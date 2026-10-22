"""Machinery shared by the FSDP2 stages (A2, B, C, D). One copy, not four.

The trainers run the same loop around a different loss: guards on the output dir,
a v2 artifact built from a checkpoint's ModelSpec, FSDP2/HSDP sharding with the fp32
alpha islands, structural lr classes, DTensor-gathered saves with a cross-rank
fingerprint, name-keyed train states, and a resume that puts optimizer moments back
onto their shards.

What stays in the trainers: the loss (A2 aligns the branch to the teacher; B adds
LoRA, an objective switch and per-modality lambdas), the trainable-set rule, the
optimizer-group layout and the metrics row. Those are the stage.
"""
import gc
import glob
import os
import re
import sys
import time

import torch
import torch.distributed as dist
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import distribute_tensor
from torch.utils.data import DataLoader, DistributedSampler

from src.checkpoints import CheckpointArtifact, load_checkpoint, save_checkpoint
from src.config import resolved_dict
from src.models.hybrid_transform import set_teacher_mode
from src.training.dataset_h3_latents import H3LatentT2VADataset, collate_single
from src.utils import run_lock
from src.utils.activation_checkpointing import split_block_checkpoints
from src.utils.lr_schedule import check_schedule_stamp
from src.utils.samplers import SkipFirst
from src.utils.device import (device_of, empty_cache, get_rng_state,
                              local_device, memory_allocated, memory_reserved,
                              set_rng_state, synchronize)


# ------------------------------------------------------------------------ phase timers
# Per-phase wall time, accumulated per step and logged alongside the loss, so a slow
# step can be attributed: data, teacher forward, student forward, backward, optimizer.
_PHASE = {}


def phase_reset():
    _PHASE.clear()


class phase:
    """Time a block into _PHASE. Synchronises so async CUDA work lands in the right
    bucket — the whole point is to tell a stall apart from compute."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        synchronize(local_device())
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        synchronize(local_device())
        _PHASE[self.name] = _PHASE.get(self.name, 0.0) + time.time() - self.t0
        return False


def phase_fields(prefix="t_"):
    """The timers as metrics-row fields (seq_len excluded from the log line)."""
    return {prefix + k: round(v, 2) for k, v in _PHASE.items()}


def phase_summary():
    return " ".join(f"{k}={v:.0f}" for k, v in sorted(_PHASE.items()) if k != "seq_len")


# ------------------------------------------------------------------- teacher forward
def set_adapters(model, enabled: bool):
    """Toggle every injected LoRA layer. The teacher is the PRISTINE released model:
    full attention AND no adapter — with the adapters left on, `teacher_mode` would
    hand back a LoRA-modified model and the alignment target would drift with training.

    Sets `_disable_adapters` directly instead of calling peft's `enable_adapters()`.
    That public method also flips `requires_grad` on the adapter weights (its docstring
    says so), and flipping requires_grad on FSDP2-managed parameters after sharding
    corrupts FSDP's per-parameter gradient-dtype bookkeeping: the very next backward
    dies with "attempting to assign a gradient with dtype BFloat16 to a tensor with
    grad_dtype Float" inside foreach_reduce. `_disable_adapters` is the flag the
    LoRA forward actually reads, and toggling it touches nothing else. A model with no
    adapters injected is a no-op."""
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module._disable_adapters = not enabled


@torch.no_grad()
def teacher_velocity(model, inputs, adapters=False):
    """The full-attention (and, with `adapters=True`, LoRA-free) prediction on the same
    noisy inputs.

    Costs one extra forward, but no second copy of the 33B base: HybridAttention keeps
    the original module as `attn.orig`, so teacher_mode (+ adapters off) IS the released
    model. Runs under no_grad, so it stores no activations and its memory is released
    before the student's backward allocates."""
    set_teacher_mode(model, True)
    if adapters:
        set_adapters(model, False)
    try:
        v, a = model(**inputs)
        return v[0].detach().float(), a[0].detach().float()
    finally:
        set_teacher_mode(model, False)
        if adapters:
            set_adapters(model, True)

        # Hand the teacher pass's blocks back before the student's forward+backward
        # allocates, so the two peaks stay sequential rather than summing.
        empty_cache(next(model.parameters()).device)


def student_forward(model, inputs, offload_activations):
    """The graph-building forward. `offload_activations`: everything autograd SAVES
    during this forward goes to pinned CPU and returns during backward. With the
    two-level checkpointing the saved set is exactly the region inputs — tens of GiB
    across the 50 blocks that otherwise sit resident through the whole forward for one
    use each in backward. The PCIe round trip is seconds against a step of minutes.
    Bitwise-neutral: tensors change address, not value."""
    if offload_activations:
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            out = model(**inputs)
    else:
        out = model(**inputs)

    return out


# ----------------------------------------------------------------------------- guards
def _rank0():
    return int(os.environ.get("RANK", "0")) == 0


def refuse_if_guarded(out_dir, done_path, ignore_stopped=False):
    """The three pre-load guards, in order. Every rank runs them (so a refusal exits
    the whole job rather than hanging the ranks that did not check); rank 0 then takes
    the run lock. Called BEFORE init_process_group.

    (a) <out_dir>/STOPPED: a deliberately stopped run stays stopped, even under an
        automatic requeue chain.
    (b) the final artifact exists: `while step < max_steps` would fall through anyway,
        but only after the load, so a requeued job would burn minutes x N nodes for
        nothing. The trainer saves unconditionally at step == max_steps.
    (c) somebody else is training into this directory. Two jobs on one output_dir
        resume from the same train_state and then overwrite each other's checkpoints
        (src/utils/run_lock.py).
    """
    stop = os.path.join(out_dir, "STOPPED")
    if os.path.exists(stop) and not ignore_stopped:
        if _rank0():
            print("=" * 78, flush=True)
            print(f"REFUSING to train: {stop} exists.", flush=True)
            with open(stop) as fh:
                for line in fh:
                    print("  " + line.rstrip(), flush=True)
            print("This run was ended on purpose; do not relaunch it automatically.",
                  flush=True)
            print("To continue it anyway, delete the file or set "
                  "training.ignore_stopped=true.", flush=True)
            print("=" * 78, flush=True)
        sys.exit(3)
    if os.path.exists(done_path):
        if _rank0():
            print(f"nothing to do: {done_path} exists, so this run already reached its "
                  f"final step. Exiting without loading the model.", flush=True)
        sys.exit(0)
    refusal = run_lock.check(out_dir)
    if refusal is not None:
        if _rank0():
            print("=" * 78, flush=True)
            print(f"REFUSING to train: {refusal}", flush=True)
            print("=" * 78, flush=True)
        sys.exit(4)
    if _rank0():
        os.makedirs(out_dir, exist_ok=True)
        run_lock.acquire(out_dir)


def load_seed_artifact(path):
    """The checkpoint whose ModelSpec decides EVERYTHING structural for this run."""
    art = load_checkpoint(path)
    if art.metadata.get("truncated_blocks"):
        raise RuntimeError(f"{path} is a truncated smoke-test artifact and must never "
                           f"seed a real run")
    return art


def print_architecture(spec_dict, label="architecture"):
    ha = spec_dict["transforms"][0]["config"]
    print(f"{label}: {ha['softmax_attention']} | {ha['linear_attention']} "
          f"| gate={ha['enable_softmax_gate']}", flush=True)


# ------------------------------------------------------------------------------ saves
def gather_named(model, select, rank):
    """Gather every parameter `select(name, param)` picks from its DTensor shard.
    Returns (state on rank 0 / {} elsewhere, fingerprint tensor, names). Non-sharded
    trainables are a bug (FSDP should own every one), and every shard participates in
    every full_tensor() -- this is a collective."""
    state, fingerprints, names = {}, [], []
    model_device = next(model.parameters()).device
    use_fp32_fingerprint = model_device.type == "npu"
    for name, param in model.named_parameters():
        t = param.data
        if hasattr(t, "full_tensor"):
            if not select(name, param):
                continue
            t = t.full_tensor()
            if rank == 0:
                state[name] = t.cpu()                    # the fp32 masters, as trained
        elif param.requires_grad:
            raise RuntimeError(f"trainable {name} is not sharded -- expected DTensor")
        fingerprints.append(
            t.detach().float().sum() if use_fp32_fingerprint
            else t.detach().double().sum()
        )
        names.append(name)
    fingerprint = torch.stack([f.to(model.device) for f in fingerprints])
    return state, fingerprint, names


def verify_fingerprint(fingerprint, names, step):
    """Refuse to save anything whose bytes differ across ranks."""
    reference = fingerprint.clone()
    dist.broadcast(reference, src=0)
    mismatched = (fingerprint != reference).nonzero().flatten().tolist()
    if mismatched:
        raise RuntimeError(
            f"rank {dist.get_rank()}: checkpoint fingerprint mismatch vs rank 0 at "
            f"step {step} ({len(mismatched)} tensors, e.g. "
            f"{[names[i] for i in mismatched[:5]]}) -- refusing to save."
        )


def save_weights_artifact(model, out_dir, filename, stage, step, rank, spec_dict, select,
                          metadata=None):
    """A v2 "weights" artifact of the parameters `select` picks -- fp32 masters, as
    trained: this file is the next stage's seed, so it keeps the precision the run had
    (inference casts on load). Saved on rank 0 after the cross-rank fingerprint check. Selection is by NAME, never requires_grad:
    a frozen softmax gate still ships its keys."""
    state, fingerprint, names = gather_named(model, select, rank)
    verify_fingerprint(fingerprint, names, step)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        receipt = {"stage": stage, "step": step}
        receipt.update(metadata or {})
        save_checkpoint(CheckpointArtifact(kind="weights", model_spec=spec_dict,
                                           weights=state, metadata=receipt), path)
        print(f"[step {step}] saved {len(state)} tensors -> {path} "
              f"(cross-rank verified: {len(names)} tensors)", flush=True)


def save_train_state(model, optimizer, weight_names, trainable_names, stage, step, epoch,
                     in_epoch, noise_generator, sigma_generator, cfg, spec_dict, out_dir,
                     rank, world, keep_states, extra_metadata=None):
    """v2 "train_state": fp32 masters (gathered) for `weight_names`, INLINE name-keyed
    optimizer state for `trainable_names`, every rank's RNG streams, the resolved
    config. Newest `keep_states` kept. `extra_metadata`: stage-private counters that
    must survive a resume (Stage-DMD's fake-score update count)."""
    params = dict(model.named_parameters())
    weights = {name: params[name].data.full_tensor() for name in weight_names}
    opt_states = {}
    for name in trainable_names:
        state = optimizer.state.get(params[name], {})
        opt_states[name] = {k: (v.full_tensor() if hasattr(v, "full_tensor") else v)
                            for k, v in state.items()}
    device = next(model.parameters()).device
    rng = {
        "torch": torch.get_rng_state(),
        device.type: get_rng_state(device),
        "noise_gen": noise_generator.get_state().cpu(),
        "sigma_gen": sigma_generator.get_state(),
    }
    all_rng = [None] * world
    dist.all_gather_object(all_rng, rng)

    if rank == 0:
        path = os.path.join(out_dir, f"train_state_step{step:06d}.pt")
        save_checkpoint(CheckpointArtifact(
            kind="train_state", model_spec=spec_dict,
            weights={k: v.float().cpu() for k, v in weights.items()},
            optimizer={k: {s: (t.cpu() if torch.is_tensor(t) else t)
                           for s, t in v.items()} for k, v in opt_states.items()},
            step=step, rng_state={"per_rank": all_rng},
            resolved_training_config=resolved_dict(cfg),
            metadata={"stage": stage, "epoch": epoch, "in_epoch": in_epoch,
                      "world_size": world, **(extra_metadata or {})}), path)
        stale = sorted(glob.glob(os.path.join(out_dir, "train_state_step*.pt")))[:-keep_states]
        for old in stale:
            os.remove(old)
        print(f"[step {step}] full train state -> {path}"
              + (f" (pruned {len(stale)})" if stale else ""), flush=True)


# ----------------------------------------------------------------------------- resume
class Resume:
    """Where a run picks up: (start_step, epoch, in_epoch) and, when a full state was
    found, the artifact itself so the optimizer/RNG half can be restored after sharding."""

    def __init__(self):
        self.start_step = self.epoch = self.in_epoch = 0
        self.full_state = None


def find_resume(out_dir, weights_prefix, spec_dict, sched, model, rank):
    """Phase A of auto-resume, BEFORE sharding: the rolling train_state (preferred)
    or the newest weights artifact. Copies weights into the unsharded model; refuses
    an artifact whose ModelSpec differs from the one this run was built from."""
    r = Resume()
    states = sorted(glob.glob(os.path.join(out_dir, "train_state_step*.pt")))
    checkpoints = sorted(glob.glob(os.path.join(out_dir, f"{weights_prefix}*.pt")))
    params = dict(model.named_parameters())
    if states:
        full = load_checkpoint(states[-1])
        if full.model_spec != spec_dict:
            raise RuntimeError(f"resume state {states[-1]} carries a different ModelSpec "
                               f"than this run was built from -- refusing to mix them")
        r.full_state = full
        r.start_step = full.step
        r.epoch = full.metadata["epoch"]
        r.in_epoch = full.metadata["in_epoch"]
        sched._resume_step = r.start_step
        saved = full.resolved_training_config
        check_schedule_stamp({"lr": saved["optimizer"]["lr"],
                              "min_lr": saved["optimizer"]["min_lr"],
                              "warmup_steps": saved["optimizer"]["warmup_steps"],
                              "max_steps": saved["training"]["max_steps"]}, sched, rank)
        for key, value in full.weights.items():
            params[key].data.copy_(value)
        if rank == 0:
            print(f"auto-resumed FULL state from step {r.start_step} "
                  f"(epoch {r.epoch}, {r.in_epoch} batches in)", flush=True)
    elif checkpoints:
        newest = checkpoints[-1]
        resumed = load_checkpoint(newest)
        if resumed.model_spec != spec_dict:
            raise RuntimeError(f"{newest} carries a different ModelSpec -- refusing")
        r.start_step = r.epoch = int(re.search(r"step(\d+)\.pt", newest).group(1))
        for key, value in resumed.weights.items():
            params[key].data.copy_(value.to(params[key].dtype))
        del resumed
        if rank == 0:
            print(f"auto-resumed weights only from {newest}", flush=True)
    return r


def restore_optimizer_and_rng(resume, model, optimizer, noise_generator, sigma_generator,
                              skip_sampler, world, rank):
    """Phase B of auto-resume, AFTER sharding and optimizer construction: moments back
    onto their shards, RNG streams and the data cursor when the world size matches.
    Returns the epoch to start from (the saved one, or `start_step` as a fresh epoch
    when the world changed and the cursor is meaningless)."""
    full = resume.full_state
    if full is None:
        return resume.epoch

    params = dict(model.named_parameters())
    for name, saved_state in full.optimizer.items():
        param = params[name]
        state = optimizer.state[param] = {}
        for key, value in saved_state.items():
            if torch.is_tensor(value) and value.shape == param.data.shape:
                state[key] = distribute_tensor(value.to(torch.float32),
                                               param.data.device_mesh,
                                               param.data.placements)
            else:
                state[key] = value
    epoch = resume.epoch
    if full.metadata["world_size"] == world:
        rng = full.rng_state["per_rank"][rank]
        torch.set_rng_state(rng["torch"])
        device = next(model.parameters()).device
        set_rng_state(rng[device.type], device)
        noise_generator.set_state(rng["noise_gen"])
        sigma_generator.set_state(rng["sigma_gen"])
        skip_sampler.skip = resume.in_epoch
    else:
        epoch = resume.start_step
        if rank == 0:
            print(f"world size changed ({full.metadata['world_size']} -> {world}): "
                  f"restored weights+optimizer, reset RNG and cursor", flush=True)
    resume.full_state = None
    del full
    gc.collect()
    empty_cache(next(model.parameters()).device)
    if rank == 0:
        device = next(model.parameters()).device
        print(f"post-resume: {memory_allocated(device)/2**30:.1f} GiB allocated, "
              f"{memory_reserved(device)/2**30:.1f} GiB reserved", flush=True)
    return epoch


# --------------------------------------------------------------------------- sharding
def shard_model(model, world, shard_size, device, rank, activation_checkpointing=True,
                extra_module_paths=()):
    """attn/ff sub-regions first, then the per-block outer checkpoint, then FSDP2/HSDP.

    FrameKDAAlpha keeps fp32 PARAMETERS, not just fp32 arithmetic. param_dtype is the
    dtype the params are all-gathered to for forward/backward, so under the plain policy
    alpha's `down`/`up` reach the forward already rounded to bf16's 8 mantissa bits — and
    the fp32 island inside FrameKDAAlpha cannot undo that, it only makes the matmul run
    in fp32 on values that are already wrong. A bf16 alpha's small per-frame error is
    compounded by the scan across all frames into tens of percent on the tails of the
    retention (how much linear state survives the clip), which is not acceptable to buy
    nothing: the fp32 masters cost ~131 MB across all 50 layers. Wrapping the submodule
    FIRST makes fully_shard(block) skip these params, since a nested FSDP module owns
    its own.
    """
    if activation_checkpointing:
        regions = split_block_checkpoints(model)
        model.enable_gradient_checkpointing()
        checkpoint_desc = f"enabled ({regions} attn/ff regions + outer DiT blocks)"
    else:
        if hasattr(model, "disable_gradient_checkpointing"):
            model.disable_gradient_checkpointing()
        checkpoint_desc = "disabled"
    if rank == 0:
        print(f"activation checkpointing: {checkpoint_desc}", flush=True)

    mesh = None
    if world > shard_size:
        mesh = init_device_mesh(device_of(device).type, (world // shard_size, shard_size),
                                mesh_dim_names=("replicate", "shard"))
        if rank == 0:
            print(f"HSDP mesh: replicate={world // shard_size} x shard={shard_size}",
                  flush=True)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    mp_fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    alpha_wrapped = 0
    for block in model.transformer_blocks:
        alpha = getattr(getattr(block.attn, "linear_attention", None), "alpha", None)
        if alpha is not None:
            fully_shard(alpha, mesh=mesh, mp_policy=mp_fp32, reshard_after_forward=True)
            alpha_wrapped += 1
    if rank == 0:
        print(f"alpha submodules kept in fp32: {alpha_wrapped}", flush=True)
    for block in model.transformer_blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    for block in model.token_refiner.refiner_blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    extra_wrapped = []
    for path in dict.fromkeys(extra_module_paths):
        if (path.startswith("transformer_blocks.")
                or path.startswith("token_refiner.refiner_blocks.")):
            raise ValueError(f"extra FSDP module {path!r} is already owned by a block")
        module = model.get_submodule(path)
        if not any(param.requires_grad for param in module.parameters()):
            raise ValueError(f"extra FSDP module {path!r} has no trainable parameters")
        fully_shard(module, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        extra_wrapped.append(path)
    if rank == 0 and extra_wrapped:
        print(f"extra FSDP modules: {extra_wrapped}", flush=True)
    return model.to(device)


# -------------------------------------------------------------------------- optimizer
def build_param_groups(model, layout, base_lr):
    """AdamW groups from `layout`: a list of (name, lr_scale, select(name, param), eps)
    over the model's trainable parameters. Groups with no members are dropped. Call it
    AFTER sharding: fully_shard swaps the Parameter objects, and the structural lr-class
    map the selectors consult is keyed on identity."""
    groups = []
    for name, scale, select, eps in layout:
        params = [p for n, p in model.named_parameters()
                  if p.requires_grad and select(n, p)]
        if not params:
            continue
        g = {"params": params, "name": name, "lr_scale": scale, "lr": base_lr * scale}
        if eps is not None:
            g["eps"] = eps
        groups.append(g)
    return groups


def describe_groups(groups):
    return "optimizer groups: " + ", ".join(
        f"{g['name']} {sum(p.numel() for p in g['params'])/1e6:.1f}M @ lr x{g['lr_scale']}"
        f"{' eps ' + str(g['eps']) if 'eps' in g else ''}" for g in groups)


def clip_gradients(trainable, grad_clip):
    """Global clip; 0 disables. Returns the (gathered) pre-clip norm."""
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, grad_clip if grad_clip > 0 else float("inf"))
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    return grad_norm


# --------------------------------------------------------------------------------- data
def make_loader(index_file, num_workers, world, rank, seed, text_only=False):
    """Dataset + DistributedSampler behind a SkipFirst (data-order-exact resume)."""
    dataset = H3LatentT2VADataset(index_file, text_only=text_only)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=seed)
    skip_sampler = SkipFirst(sampler)
    loader = DataLoader(dataset, batch_size=1, sampler=skip_sampler,
                        num_workers=num_workers, collate_fn=collate_single,
                        pin_memory=True, drop_last=True)
    return dataset, sampler, skip_sampler, loader


def make_generators(seed, rank, device):
    """Per-rank noise (device) and sigma (CPU) streams; the model init itself uses the
    SAME seed on every rank (torch.manual_seed by the caller) so FSDP2, which does not
    broadcast, shards identical values."""
    noise_generator = torch.Generator(device=device).manual_seed(seed * 100003 + rank)
    sigma_generator = torch.Generator().manual_seed(seed * 100019 + rank)
    return noise_generator, sigma_generator
