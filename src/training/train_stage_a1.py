"""Stage A1: per-layer teacher alignment of the freshly-built hybrid.

  config   load_config(StageA1Config): YAML + dotlist, --print-config, --validate-only.
  build    resolve_model_spec -> build_model(spec). The teacher backend and the
           smoke-test truncation apply AFTER the build: runtime, not architecture.
  train    AlignHook (one layer's graph at a time), ZeRO-1 owner sharding of the
           optimizer moments, per-layer clipping.
  save     hybrid_stepNNNNNN.pt is a v2 "weights" artifact carrying the full
           ModelSpec. Nothing else: A1 is short and is assumed to finish in one
           shot, so there is no train state and no resume; a finished run's output
           dir refuses to retrain.

Run:
    torchrun --standalone --nproc_per_node=8 src/training/train_stage_a1.py \
        --config configs/training/stage_a1_c1_vdn_anchor.yaml \
        [optimizer.lr=1e-4 training.max_steps=200 ...]
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

from src.training.dataset_h3_latents import H3LatentT2VADataset, collate_single
from src.models.softmax_attention import FlexFA4Processor
from src.utils.distributed import broadcast_and_verify_init
from src.utils.lr_schedule import lr_at

from src.checkpoints import CheckpointArtifact
from src.checkpoints import save_checkpoint
from src.config import load_config
from src.config.common import validate_enums
from src.config.stage_a1 import StageA1Config
from src.models.factory import build_model, load_model_weights, resolve_model_spec
from src.models.hybrid_transform import (hybrid_new_parameters, iter_hybrids, set_layout, set_softmax_backend, set_teacher_mode)
from src.utils.lr_classes import lr_class_map
from src.training.t2va_batch import pack_noisy_batch
from src.utils import run_lock
from src.utils.device import (autocast, distributed_backend, local_device,
                              max_memory_allocated, memory_allocated,
                              reset_peak_memory_stats, set_device,
                              set_memory_fraction)


class AlignHook:
    """Forward hook on a HybridAttention running in teacher_mode.

    The trunk (under no_grad) has just produced the TEACHER output for this layer; the
    hook re-runs the layer as the hybrid student from the detached input and backwards
    the per-layer MSE immediately, so at most one layer's autograd graph exists at any
    moment. The frozen base + detached input mean the graph starts at the first branch
    parameter: the windowed flex forward itself records nothing.

    `scale` is 1/grad_accum — gradients from the micro-batches of one optimizer step
    accumulate in .grad exactly as the main trainer's (loss/N).backward() does.
    """

    def __init__(self, hybrid, index):
        self.hybrid = hybrid
        self.index = index
        self.scale = 1.0
        self.loss = torch.zeros(())      # detached fp32 scalars, refreshed every micro
        self.rel = torch.zeros(())

    def __call__(self, module, args, kwargs, output):
        x = (args[0] if args else kwargs["hidden_states"]).detach()
        rotary_emb = args[1] if len(args) > 1 else kwargs.get("rotary_emb")
        teacher = output.detach()[0]                               # [S, hidden] bf16
        teacher_sq = teacher.float().pow(2).sum().clamp_min(1e-30)
        with torch.enable_grad():
            with autocast(x.device, dtype=torch.bfloat16):
                student = self.hybrid._hybrid_forward(x[0], rotary_emb)
            diff = student.float() - teacher.float()
            del student

            # NORMALIZED objective: sum (s-t)^2 / sum t^2, i.e. rel^2. The raw MSE
            # rides the teacher's output scale, which varies by orders of magnitude
            # across layers and would leave the grad clip doing all the stepping;
            # normalizing puts every layer's loss AND gradients on the same relative
            # scale, so the clip becomes a genuine outlier guard.
            mse = diff.pow(2).mean()
            loss = diff.pow(2).sum() / teacher_sq
            del diff
            (loss * self.scale).backward()
        self.loss = mse.detach()
        self.rel = loss.detach().sqrt()
        if os.environ.get("H3_DEBUG_MEM") and dist.get_rank() == 0:
            device = x.device
            print(f"    block {self.index:2d}: alloc={memory_allocated(device)/2**30:.1f}"
                  f" peak={max_memory_allocated(device)/2**30:.1f}GiB", flush=True)
        return None                                                # trunk keeps teacher



def save_weights(per_block, step, cfg, spec_dict, rank,
                 all_per_block=None, truncated=0):
    """hybrid_stepNNNNNN.pt (v2 "weights", ModelSpec aboard). No train state: A1 is
    short and is assumed to finish in one shot -- no optimizer shards, no rng, no
    resume. Refuses to save if the parameters differ across ranks."""
    out_dir = cfg.checkpoint.output_dir
    names, params = [], []
    for _, named in (all_per_block or per_block):
        for name, p in named:
            names.append(name)
            params.append(p)
    fingerprint = torch.stack([
        p.detach().float().sum() if p.device.type == "npu" else p.detach().double().sum()
        for p in params
    ])
    reference = fingerprint.clone()
    dist.broadcast(reference, src=0)
    mismatched = (fingerprint != reference).nonzero().flatten().tolist()
    if mismatched:
        raise RuntimeError(
            f"rank {rank}: checkpoint fingerprint mismatch vs rank 0 at step {step} "
            f"({len(mismatched)} tensors, e.g. {[names[i] for i in mismatched[:5]]}) "
            "-- refusing to save."
        )

    if rank == 0:
        weights = {name: p.detach().cpu() for name, p in zip(names, params)}   # fp32 masters
        meta = {"stage": "a1", "step": step}
        if truncated:
            meta["truncated_blocks"] = truncated   # smoke artifact: unusable downstream
        path = os.path.join(out_dir, f"hybrid_step{step:06d}.pt")
        save_checkpoint(CheckpointArtifact(kind="weights", model_spec=spec_dict,
                                   weights=weights, metadata=meta), path)
        print(f"[step {step}] saved {path} "
              f"(cross-rank verified: {len(names)} tensors)", flush=True)
    dist.barrier()


def main():
    cfg = load_config(StageA1Config, extra_validators=[validate_enums])
    sched = SimpleNamespace(lr=cfg.optimizer.lr, min_lr=cfg.optimizer.min_lr,
                            warmup_steps=cfg.optimizer.warmup_steps,
                            max_steps=cfg.training.max_steps)
    out_dir = cfg.checkpoint.output_dir
    grad_accum = cfg.training.gradient_accumulation_steps

    refusal = run_lock.check(out_dir)
    if refusal is not None:
        if int(os.environ.get("RANK", "0")) == 0:
            print("=" * 78, flush=True)
            print(f"REFUSING to train: {refusal}", flush=True)
            print("=" * 78, flush=True)
        sys.exit(4)
    if int(os.environ.get("RANK", "0")) == 0:
        os.makedirs(out_dir, exist_ok=True)
        run_lock.acquire(out_dir)

    device = local_device()
    set_device(device)
    dist.init_process_group(distributed_backend(device))
    rank, world = dist.get_rank(), dist.get_world_size()

    # A1 runs close to the card's capacity on 141 GB GPUs: each layer's backward
    # recompute transiently holds one projection's conv copies + SiLU/L2Norm saves,
    # and the per-layer fp32 grads accumulate across all 50 blocks by design (the
    # all-reduce + ZeRO-1 step happen after the batch). 0.96 leaves a few GiB for
    # driver/NCCL variance.
    set_memory_fraction(0.96, device)

    # SAME seed on every rank: the branch inits draw from the global RNG, and ranks must
    # agree before broadcast_and_verify_init proves they do. Noise/sigma streams are
    # per-rank generators below.
    torch.manual_seed(cfg.training.seed)

    # -------- build seam: config -> spec -> model --------
    base_config = MiniMaxH3Transformer3DModel.load_config(
        cfg.model.base.source, subfolder=cfg.model.base.subfolder)
    spec = resolve_model_spec(cfg.model, dict(base_config))
    spec_dict = spec.to_dict()
    model = build_model(spec, device="cpu", base_source=cfg.model.base.source)
    set_softmax_backend(model, cfg.runtime.kernels.softmax_backend, device=device)
    model.requires_grad_(False)
    truncated = int(cfg.training.truncate_blocks)
    if truncated:
        model.transformer_blocks = model.transformer_blocks[:truncated]
        if rank == 0:
            print(f"SMOKE TEST: truncated to {truncated} blocks -- "
                  "checkpoints will be stamped unusable for real runs", flush=True)
    if cfg.training.teacher_backend == "flex_fa4" and device.type == "cuda":
        for module in model.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.set_processor(FlexFA4Processor())
    elif cfg.training.teacher_backend == "flex_fa4" and device.type == "npu":
        if rank == 0:
            print("teacher_backend=flex_fa4 requested on NPU; using the native "
                  "Diffusers attention processor", flush=True)
    elif cfg.training.teacher_backend:
        model.set_attention_backend(cfg.training.teacher_backend)
    hybrids = list(iter_hybrids(model))
    for attn in hybrids:
        # A1 has no per-block gradient checkpointing; recompute the feature chain in
        # backward instead of saving it (see BidirectionalLinearBranch).
        attn.linear_attention.checkpoint_features = True
    set_teacher_mode(model, True)
    per_block = hybrid_new_parameters(model)

    # freeze_softmax_gate holds the gate at its 0.99 init; architecture and checkpoint
    # keys are unchanged.
    classes = lr_class_map(model)                      # structural
    branch_params, frozen, trainable_per_block = [], 0, []
    for i, named in per_block:
        keep = []
        for name, p in named:
            p.data = p.data.to(torch.float32)   # amp masters; compute is autocast bf16
            if cfg.training.freeze_softmax_gate and ".softmax_gate." in name:
                p.requires_grad_(False)
                frozen += p.numel()
                continue
            p.requires_grad_(True)
            branch_params.append(p)
            keep.append((name, p))
        trainable_per_block.append((i, keep))
    per_block, all_per_block = trainable_per_block, per_block
    if rank == 0:
        n = sum(p.numel() for p in branch_params)
        ha = spec.transforms[0].config
        note = (f", softmax gate FROZEN at 0.99 ({frozen / 1e6:.1f}M held)"
                if cfg.training.freeze_softmax_gate else "")
        print(f"hybrid: {len(per_block)} blocks, {n / 1e6:.1f}M branch params, "
              f"window {ha['softmax_attention']}, linear {ha['linear_attention']}{note}",
              flush=True)

    # No resume seam: A1 saves no train state -- it is short and assumed to finish
    # in one shot. A finished run is still guarded: refuse to
    # retrain over an output dir that already holds the final artifact.
    final_path = os.path.join(out_dir, f"hybrid_step{cfg.training.max_steps:06d}.pt")
    if os.path.exists(final_path):
        if rank == 0:
            print(f"{final_path} already exists -- stage A1 is done, exiting.",
                  flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    broadcast_and_verify_init(
        model, [p for _, named in all_per_block for _, p in named], device)
    if rank == 0:
        print("init broadcast + cross-rank verification passed", flush=True)

    model = model.to(device)

    # `owner` is keyed by POSITION in per_block, so the group's layer key must be the
    # position too, not the block index off the tuple. They are equal today (every block
    # is appended, even one whose params are all frozen) but that is an invariant of
    # hybrid_new_parameters, not something this loop should assume -- mis-keying it would
    # shard the optimiser wrong and silently corrupt the update.
    assert [i for i, _ in per_block] == list(range(len(per_block))), \
        "per_block is no longer position-indexed; the ZeRO owner map needs updating"
    groups = []
    for pos, (i, named) in enumerate(per_block):
        for cls, scale in (("big", cfg.optimizer.big_lr_scale), ("small", cfg.optimizer.small_lr_scale)):
            params = [p for nm, p in named if classes.get(id(p), "big") == cls]
            if not params:
                continue
            g = {"params": params, "lr": cfg.optimizer.lr * scale, "name": f"block{i}.{cls}",
                 "layer": pos, "lr_scale": scale}
            if cls == "small":
                g["eps"] = cfg.optimizer.small_eps
            groups.append(g)

    # Clip over each LAYER's full parameter set, not per group: splitting a layer in two
    # and clipping the halves independently would silently change what --grad_clip means.
    clip_sets = [[p for g in groups if g["layer"] == pos for p in g["params"]]
                 for pos in range(len(per_block))]
    optimizer = torch.optim.AdamW(groups, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    if rank == 0:
        for cls in ("big", "small"):
            tot = sum(p.numel() for g in groups if g["name"].endswith(cls)
                      for p in g["params"])
            scale = cfg.optimizer.big_lr_scale if cls == "big" else cfg.optimizer.small_lr_scale
            print(f"optimizer: {cls} {tot/1e6:.1f}M @ lr x{scale}"
                  + (f" eps {cfg.optimizer.small_eps:g}" if cls == "small" else ""), flush=True)

    # ZeRO-1 over layers: each rank keeps AdamW moments ONLY for the layers it owns and
    # broadcasts their updated params after the step. Full moments are 2 x 4 bytes over
    # ~2.1B branch params = ~16 GiB per GPU, which does not fit next to the 62 GiB base
    # replica; owner-sharded they are ~2 GiB. Grads are still all-reduced (identical on
    # every rank), so the clip norms and the fingerprints stay rank-invariant.
    owner = {idx: idx % world for idx in range(len(per_block))}

    hooks = []
    for idx, ((i, _), hybrid) in enumerate(zip(per_block, hybrids)):
        hook = AlignHook(hybrid, i)
        model.transformer_blocks[i].attn.register_forward_hook(hook, with_kwargs=True)
        hooks.append(hook)

    dataset = H3LatentT2VADataset(cfg.data.index_file)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg.training.seed)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                        num_workers=cfg.data.num_workers,
                        collate_fn=collate_single, pin_memory=True, drop_last=True)
    if rank == 0:
        print(f"dataset: {len(dataset)} clips, {len(loader)} steps/epoch/rank, "
              f"global batch {world * grad_accum}", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "metrics.jsonl")

    noise_generator = torch.Generator(device=device).manual_seed(
        cfg.training.seed * 100003 + rank)
    sigma_generator = torch.Generator().manual_seed(cfg.training.seed * 100019 + rank)

    num_layers = len(hooks)
    step, epoch, in_epoch, micro = 0, 0, 0, 0
    acc = torch.zeros(2, num_layers, device=device)
    t_log = time.time()
    model.eval()
    early_saves = set(cfg.checkpoint.early_saves)
    if 0 in early_saves and step == 0:
        save_weights(per_block, 0, cfg, spec_dict, rank,
                     all_per_block=all_per_block, truncated=truncated)
    while step < cfg.training.max_steps:
        sampler.set_epoch(epoch)
        for sample in loader:
            in_epoch += 1
            batch = pack_noisy_batch(sample, device, noise_generator, sigma_generator)
            set_layout(model, batch["layout"])
            for hook in hooks:
                hook.scale = 1.0 / grad_accum
            with torch.no_grad():
                model(**batch["inputs"])
            acc[0] += torch.stack([h.loss for h in hooks])
            acc[1] += torch.stack([h.rel for h in hooks])
            micro += 1
            if micro % grad_accum != 0:
                continue

            for p in branch_params:
                if p.grad is None:                # a layer whose loss was exactly 0
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            lr = lr_at(step, sched)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_scale"]
            grad_norms = torch.stack([
                torch.nn.utils.clip_grad_norm_(params, cfg.optimizer.grad_clip)
                for params in clip_sets])
            for group in optimizer.param_groups:
                if owner[group["layer"]] != rank:
                    for p in group["params"]:
                        p.grad = None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for idx, (_, named) in enumerate(per_block):
                for _, p in named:
                    dist.broadcast(p.data, src=owner[idx])
            step += 1

            avg = acc / grad_accum
            acc = torch.zeros_like(acc)
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            mse, rel = avg[0], avg[1]
            if rank == 0:
                dt = time.time() - t_log
                peak = max_memory_allocated(device) / 2**30
                worst = int(rel.argmax())
                print(f"[step {step}/{cfg.training.max_steps}] align_mse={mse.mean():.5f} "
                      f"rel={rel.mean():.4f} (worst block {worst}: {rel[worst]:.4f}) "
                      f"grad_norm mean={grad_norms.mean():.4f} max={grad_norms.max():.4f} "
                      f"lr={lr:.2e} {dt:.1f}s/step peak={peak:.1f}GiB", flush=True)
                if step % cfg.training.per_layer_every == 0 or step == 1:
                    for i in range(0, num_layers, 10):
                        row = " ".join(f"{j:2d}:{rel[j]:.3f}"
                                       for j in range(i, min(i + 10, num_layers)))
                        print(f"  rel {row}", flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "align_mse": round(mse.mean().item(), 6),
                        "rel_mean": round(rel.mean().item(), 6),
                        "rel_per_layer": [round(v, 5) for v in rel.tolist()],
                        "mse_per_layer": [round(v, 4) for v in mse.tolist()],
                        "grad_norm_mean": round(grad_norms.mean().item(), 6),
                        "grad_norm_max": round(grad_norms.max().item(), 6),
                        "grad_norm_per_layer": [round(v, 5) for v in grad_norms.tolist()],
                        "lr": lr, "s_per_step": round(dt, 2),
                        "peak_gib": round(peak, 2)}) + "\n")
            t_log = time.time()
            reset_peak_memory_stats(device)
            periodic = (step % cfg.checkpoint.save_every == 0
                        or step == cfg.training.max_steps)
            if periodic or step in early_saves:
                save_weights(per_block, step, cfg, spec_dict, rank,
                             all_per_block=all_per_block, truncated=truncated)
            if step >= cfg.training.max_steps:
                break
        epoch += 1
        in_epoch = 0

    dist.barrier()
    if rank == 0:
        print("stage A1 done", flush=True)
        run_lock.release(out_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
