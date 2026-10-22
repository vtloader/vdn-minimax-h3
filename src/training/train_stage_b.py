"""Stage B: LoRA + linear-branch fine-tune against the frozen teacher, architecture
inherited from the A1/A2 checkpoint, adapters stamped into the ModelSpec.

    torchrun --standalone --nproc_per_node=8 src/training/train_stage_b.py \\
        --config configs/training/stage_b_c1_vdn_anchor.yaml \\
        [initialization.checkpoint=... optimizer.lr=1e-4 ...]

The FSDP2/HSDP machinery is src/training/fsdp_stage.py, shared with A2. What is B's
own: the adapter rule (below), the objective switch with per-modality lambdas and the
7-metric accounting (excess_vs_teacher), the lora / branch_big / branch_small optimizer
layout, and the trainable set (transform parameters + every LoRA tensor).

  B fresh   requires cfg.lora; injects peft; stamps an AdapterSpec into the ModelSpec
            all its artifacts carry.
  B resume  takes rank/alpha/targets FROM the checkpoint's adapter spec; a cfg.lora that
            disagrees is refused before the model exists.
"""
import gc
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import load_config
from src.config.stage_b import StageBConfig
from src.models.factory import build_model, inject_adapters, load_model_weights
from src.models.hybrid_transform import (is_transform_parameter, set_layout,
                                         set_softmax_backend)
from src.models.model_spec import ModelSpec
from src.training import fsdp_stage as fs
from src.training.fsdp_stage import _PHASE, phase
from src.training.t2va_batch import pack_noisy_batch
from src.utils import run_lock
from src.utils.distributed import broadcast_and_verify_init
from src.utils.lr_classes import lr_class_map
from src.utils.lr_schedule import lr_at
from src.utils.device import (distributed_backend, empty_cache, local_device,
                              max_memory_allocated, reset_peak_memory_stats,
                              set_device, set_memory_fraction)

STAGE = "b"
WEIGHTS_PREFIX = "hybrid_lora_step"


def is_lora(name):
    return "lora_" in name


def compute_loss(model, sample, device, noise_generator, sigma_generator, audio_loss_weight,
                 offload_activations, objective="diffusion", lam_video=0.0, lam_audio=0.0):
    with phase("data"):
        batch = pack_noisy_batch(sample, device, noise_generator, sigma_generator)
        set_layout(model, batch["layout"])
    _PHASE["seq_len"] = float(batch["layout"].seq_len)   # last micro-batch's length

    # `teacher` regresses on the released model's own prediction instead of the data.
    # The diffusion target x0 - noise is a ONE-SAMPLE draw whose irreducible variance is
    # most of the ~0.5 loss floor; the teacher's output approximates E[v | x_t], the
    # quantity that loss is trying to estimate, so the target is variance-free by
    # comparison. It also keeps Stage-A and Stage-B in one objective family (per-layer
    # alignment -> end-to-end alignment) instead of switching objectives at the handoff.
    if objective == "teacher":
        with phase("teacher_fwd"):
            target_v, target_a = fs.teacher_velocity(model, batch["inputs"], adapters=True)
    else:
        target_v, target_a = batch["target_video_rows"], batch["target_audio_rows"]

    with phase("student_fwd"):
        velocity_v, velocity_a = fs.student_forward(model, batch["inputs"],
                                                    offload_activations)
    pred_v, pred_a = velocity_v[0].float(), velocity_a[0].float()

    loss_video = torch.nn.functional.mse_loss(pred_v, target_v)
    loss_audio = torch.nn.functional.mse_loss(pred_a, target_a)
    loss_teacher_only = loss_video + audio_loss_weight * loss_audio

    # The data-target MSEs. Computed ONCE and used for both the (optional) live mixing
    # term and the logged shared metric, so turning lambda on costs no extra work.
    mix = objective == "teacher" and (lam_video > 0 or lam_audio > 0)
    with torch.set_grad_enabled(mix):
        data_v = torch.nn.functional.mse_loss(pred_v, batch["target_video_rows"])
        data_a = torch.nn.functional.mse_loss(pred_a, batch["target_audio_rows"])

    # loss = teacher + lambda * diffusion, with lambda PER MODALITY. One global lambda is
    # wrong here because the two modalities have very different noise structure: the
    # student-vs-teacher error on video is comparable to the student-vs-data loss,
    # while on audio it is an order of magnitude smaller (windowing only touches video;
    # audio rows stay dense as globals, so the student already reproduces the released
    # model's audio almost exactly) and the data loss there is almost entirely sampling
    # variance. A single lambda would be a mild nudge on video and a takeover on audio.
    #
    # What lambda is FOR: not correcting the teacher's bias -- this pipeline is trying to
    # REPRODUCE the released model at lower cost, so that bias is not an error we want
    # removed. It is a regulariser against overfitting the teacher's single-step,
    # teacher-forced behaviour, which is the objective's known blind spot (nothing here
    # ever sees the 50-step rollout). Judge it on renders, not on the loss.
    loss = loss_teacher_only
    if mix:
        loss = loss + lam_video * data_v + audio_loss_weight * lam_audio * data_a

    with torch.no_grad():
        if objective == "teacher":
            loss_ref = data_v.detach() + audio_loss_weight * data_a.detach()

            # The teacher's OWN data-target loss: the floor under loss_ref. The ~90% of
            # it that is sampling variance is the SAME draw for teacher and student, so
            # it cancels in the difference, and `loss_ref - loss_diffusion_teacher` is
            # the only cross-objective-comparable quality number we have. The raw
            # loss_ref is not: it IS the diffusion arm's training objective.
            loss_ref_teacher = (torch.nn.functional.mse_loss(target_v, batch["target_video_rows"])
                                + audio_loss_weight
                                * torch.nn.functional.mse_loss(target_a, batch["target_audio_rows"]))
        else:
            loss_ref = loss.detach()
            loss_ref_teacher = torch.zeros_like(loss_ref)

    # Zeroed THROUGH the graph so FSDP's reduction still sees every trainable parameter.
    if not torch.isfinite(loss):
        print(f"non-finite loss at t_v={batch['t_v']:.4f}; zeroing this step", flush=True)
        loss = (velocity_v.float().sum() + velocity_a.float().sum()) * 0.0
        loss_video = loss_audio = loss_ref = loss_ref_teacher = loss.detach()
        loss_teacher_only = loss.detach()

    return (loss, loss_video.detach(), loss_audio.detach(), loss_ref, batch["t_v"],
            loss_teacher_only.detach(), loss_ref_teacher)


def resolve_lora_model_spec(source_model_spec, cfg, rank):
    """Apply the Stage-B LoRA rule and return a resolved model-spec mapping.

    Fresh (spec has no adapters): cfg.lora is required; its spec is stamped so every
    checkpoint this run writes carries it. Seeded/resumed (spec HAS adapters): the
    checkpoint is the authority; a cfg.lora that disagrees is refused."""
    adapters = source_model_spec.get("adapters") or []

    if adapters:                                     # seeded / resumed: checkpoint rules
        checkpoint_lora_spec = adapters[0]["config"]
        if cfg.lora is not None:
            if cfg.lora.rank <= 0 or cfg.lora.alpha <= 0:
                raise RuntimeError("LoRA rank and alpha must both be explicit positive integers")
            want = dict(rank=cfg.lora.rank, alpha=cfg.lora.alpha,
                        targets=sorted(cfg.lora.targets))
            have = dict(rank=checkpoint_lora_spec["rank"],
                        alpha=checkpoint_lora_spec["alpha"],
                        targets=sorted(checkpoint_lora_spec["targets"]))
            if want != have:
                raise RuntimeError(
                    f"the checkpoint's adapter spec {have} disagrees with the YAML's "
                    f"{want}; B resume takes its LoRA architecture FROM the checkpoint "
                    f"-- drop the lora section or fix the pointer")
        resolved_model_spec = source_model_spec
    else:                                            # fresh: YAML rules, stamp it
        if cfg.lora is None:
            raise RuntimeError("B fresh needs a lora section (rank/alpha/targets); the "
                               "seed checkpoint carries no adapter spec")
        if cfg.lora.rank <= 0 or cfg.lora.alpha <= 0:
            raise RuntimeError("LoRA rank and alpha must both be explicit positive integers")
        lora_spec = dict(rank=cfg.lora.rank, alpha=cfg.lora.alpha,
                         targets=sorted(cfg.lora.targets))
        resolved_model_spec = dict(source_model_spec)
        resolved_model_spec["adapters"] = [
            dict(type="lora", version=1, config=lora_spec)
        ]
        ModelSpec.from_dict(resolved_model_spec)     # validates before anything trains
        if rank == 0:
            print(f"adapter spec stamped: lora rank {lora_spec['rank']}, "
                  f"{len(lora_spec['targets'])} target patterns", flush=True)

    return resolved_model_spec


def run(cfg, stage=STAGE, model_spec_adapter=None):
    lr_schedule = SimpleNamespace(lr=cfg.optimizer.lr, min_lr=cfg.optimizer.min_lr,
                                  warmup_steps=cfg.optimizer.warmup_steps,
                                  max_steps=cfg.training.max_steps)
    output_dir = cfg.checkpoint.output_dir
    gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    max_steps = cfg.training.max_steps

    final_weights_path = os.path.join(
        output_dir, f"{WEIGHTS_PREFIX}{max_steps:06d}.pt")
    fs.refuse_if_guarded(
        output_dir, final_weights_path, cfg.training.ignore_stopped)

    device = local_device()
    set_device(device)
    dist.init_process_group(distributed_backend(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    set_memory_fraction(cfg.distributed.memory_fraction, device)
    torch.manual_seed(cfg.training.seed)   # same-seed init; FSDP2 does not broadcast

    # The initialization checkpoint carries the model recipe and the learned deltas;
    # the released 33B base itself is loaded separately by build_model().
    initialization_checkpoint = fs.load_seed_artifact(cfg.initialization.checkpoint)
    resolved_model_spec = resolve_lora_model_spec(
        initialization_checkpoint.model_spec, cfg, rank)
    if model_spec_adapter is not None:
        resolved_model_spec = model_spec_adapter(
            resolved_model_spec, cfg, rank, initialization_checkpoint)

    lora_spec = resolved_model_spec["adapters"][0]["config"]
    if rank == 0:
        fs.print_architecture(resolved_model_spec)

    # Rebuild released base + HybridAttention, then restore every non-LoRA transform
    # tensor: Linear Branch, gates, norms, short conv, and to_out_linear.
    model = build_model(
        resolved_model_spec, device="cpu",
        base_source=cfg.initialization.base_source)
    set_softmax_backend(model, cfg.runtime.kernels.softmax_backend, device=device)
    transform_weights = {
        name: tensor
        for name, tensor in initialization_checkpoint.weights.items()
        if not is_lora(name)
    }
    loaded_transform_tensors = load_model_weights(model, transform_weights)
    model.requires_grad_(False)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"DiT built from spec + {loaded_transform_tensors} transform tensors: "
              f"{n / 1e9:.1f}B params",
              flush=True)

    # LoRA parameters do not exist until PEFT injects the adapter modules. Stage B's
    # A2 seed has no LoRA tensors; Stage C restores them from the Stage-B checkpoint.
    model = inject_adapters(model, resolved_model_spec)
    lora_weights = {
        name: tensor
        for name, tensor in initialization_checkpoint.weights.items()
        if is_lora(name)
    }
    unsharded_parameters_by_name = None
    if lora_weights:
        unsharded_parameters_by_name = dict(model.named_parameters())
        missing = [name for name in lora_weights
                   if name not in unsharded_parameters_by_name]
        if missing:
            raise RuntimeError(f"{len(missing)} LoRA keys have no module after "
                               f"injection, e.g. {missing[:3]}")
        for name, tensor in lora_weights.items():
            parameter = unsharded_parameters_by_name[name]
            parameter.data.copy_(tensor.to(parameter.dtype))
        if rank == 0:
            print(f"restored {len(lora_weights)} LoRA tensors from the initialization "
                  "checkpoint", flush=True)
    del initialization_checkpoint, transform_weights, lora_weights

    # Everything begins frozen. The only FP32 trainable masters are LoRA and the
    # Hybrid Transform parameters introduced around the frozen base attention.
    unsharded_trainable_parameters = []
    num_lora_parameters = 0
    num_transform_parameters = 0
    for name, param in model.named_parameters():
        lora, branch = is_lora(name), is_transform_parameter(name)
        if lora or branch:
            param.data = param.data.to(torch.float32)
            param.requires_grad_(True)
            unsharded_trainable_parameters.append(param)
            num_lora_parameters += param.numel() * lora
            num_transform_parameters += param.numel() * branch

    num_lora_modules = sum(
        1 for name, _ in model.named_modules() if name.endswith("lora_A"))
    if num_lora_modules == 0:
        raise RuntimeError("LoRA matched no modules — the target names do not exist in this port.")
    if rank == 0:
        print(f"trainables: LoRA {num_lora_modules} modules / "
              f"{num_lora_parameters / 1e6:.1f}M params (rank {lora_spec['rank']}), "
              f"branch {num_transform_parameters / 1e6:.1f}M params", flush=True)

    # The initialization checkpoint seeds a new run. Auto-resume is a separate overlay
    # from this stage's output directory and may restore newer weights and optimizer state.
    resume_state = fs.Resume()
    if cfg.training.auto_resume:
        resume_state = fs.find_resume(
            output_dir, WEIGHTS_PREFIX, resolved_model_spec, lr_schedule, model, rank)

    # FSDP2 does not perform rank-0 initialization broadcast, so verify identical
    # unsharded parameters on every rank before replacing them with DTensor shards.
    broadcast_and_verify_init(model, unsharded_trainable_parameters, device)
    if rank == 0:
        print("init broadcast + cross-rank verification passed", flush=True)

    # fully_shard replaces each registered Parameter with a DTensor Parameter. If a
    # caller-owned container still references the old objects while model.to(device)
    # runs, those old objects become a complete unsharded CUDA replica (tens of GiB
    # next to the correct local shards). Their only work (copy + broadcast) is
    # finished, so drop them before FSDP starts rather than relying on cleanup
    # afterwards.
    del unsharded_trainable_parameters
    unsharded_parameters_by_name = None
    del param
    gc.collect()

    model = fs.shard_model(
        model, world, cfg.distributed.shard_size, device, rank,
        activation_checkpointing=cfg.distributed.activation_checkpointing)

    # Clear temporary wrapping objects and return unused allocator blocks before the
    # first all-gather. This is not the replica fix above (allocated memory is already
    # shard-sized); it makes the physical baseline deterministic as well.
    gc.collect()
    empty_cache(device)
    if rank == 0:
        print(f"saved-tensor CPU offload: {cfg.distributed.offload_activations}", flush=True)

    # lr classes AFTER sharding (identity-keyed); lora by peft's name.
    parameter_lr_classes = lr_class_map(model)
    optimizer_groups = fs.build_param_groups(model, [
        ("lora", 1.0, lambda n, p: is_lora(n), None),
        ("branch_big", cfg.optimizer.branch_lr_scale,
         lambda n, p: not is_lora(n)
         and parameter_lr_classes.get(id(p), "big") == "big", None),
        ("branch_small", cfg.optimizer.small_lr_scale,
         lambda n, p: not is_lora(n)
         and parameter_lr_classes.get(id(p), "big") == "small",
         cfg.optimizer.small_eps),
    ], cfg.optimizer.lr)
    if rank == 0:
        print(fs.describe_groups(optimizer_groups), flush=True)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=cfg.optimizer.lr,
                                  weight_decay=cfg.optimizer.weight_decay)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    # Build the distributed data stream, then restore optimizer/RNG state only after
    # FSDP has established the final sharded Parameter objects.
    dataset, distributed_sampler, resumable_sampler, data_loader = fs.make_loader(
        cfg.data.index_file, cfg.data.num_workers, world, rank, cfg.training.seed)
    if rank == 0:
        print(f"dataset: {len(dataset)} clips, {len(data_loader)} steps/epoch/rank, "
              f"global batch {world * gradient_accumulation_steps}", flush=True)
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
    noise_generator, sigma_generator = fs.make_generators(cfg.training.seed, rank, device)
    epoch = fs.restore_optimizer_and_rng(
        resume_state, model, optimizer, noise_generator, sigma_generator,
        resumable_sampler, world, rank)

    # Lightweight weights checkpoints carry the model recipe plus Branch/LoRA. Full
    # train-state checkpoints additionally carry AdamW moments, RNG, and data position.
    def save_weights(step):
        # every trainable: LoRA pairs + the branch, by name
        fs.save_weights_artifact(
            model, output_dir, f"{WEIGHTS_PREFIX}{step:06d}.pt", stage, step, rank,
            resolved_model_spec,
            select=lambda n, p: is_lora(n) or is_transform_parameter(n))

    def save_state(step, epoch, in_epoch):
        fs.save_train_state(model, optimizer, trainable_names, trainable_names, stage, step,
                            epoch, in_epoch, noise_generator, sigma_generator, cfg,
                            resolved_model_spec, output_dir, rank, world,
                            cfg.checkpoint.keep_states)

    # From here onward Stage B and Stage C share the exact same optimization loop.
    model.train()
    step = resume_state.start_step
    in_epoch = resume_state.in_epoch
    micro_step = 0
    early_saves = set(cfg.checkpoint.early_saves)

    # Step 0 gets the trainables file but NOT a full train_state: the optimizer is still
    # empty, so a resume from it would be a resume from scratch, and it would cost a
    # large write. What it IS good for is rendering — it is the Stage-A model as seen
    # through Stage-B's own conversion + merge path, the anchor the later steps drift from.
    if 0 in early_saves and step == 0:
        save_weights(0)
    # loss, video, audio, loss_diffusion, t_v, loss_teacher_only, loss_diffusion_teacher
    acc = torch.zeros(7, device=device)
    t_log = time.time()
    while step < max_steps:
        distributed_sampler.set_epoch(epoch)
        for sample in data_loader:
            in_epoch += 1
            (loss, loss_video, loss_audio, loss_ref, t_v,
             loss_teacher_only, loss_ref_teacher) = compute_loss(
                model, sample, device, noise_generator, sigma_generator,
                cfg.training.audio_loss_weight, cfg.distributed.offload_activations,
                cfg.training.objective, cfg.training.lambda_diffusion_video,
                cfg.training.lambda_diffusion_audio)
            with phase("backward"):
                (loss / gradient_accumulation_steps).backward()
            acc += torch.stack([loss.detach(), loss_video, loss_audio, loss_ref,
                                torch.tensor(t_v, device=device),
                                loss_teacher_only, loss_ref_teacher])
            micro_step += 1
            if micro_step % gradient_accumulation_steps != 0:
                continue

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, lr_schedule) * group["lr_scale"]
            grad_norm = fs.clip_gradients(trainable_parameters, cfg.optimizer.grad_clip)
            with phase("optim"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1

            avg = acc / gradient_accumulation_steps
            acc = torch.zeros_like(acc)
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            if rank == 0:
                dt = time.time() - t_log
                peak = max_memory_allocated(device) / 2**30
                ref = "" if cfg.training.objective == "diffusion" else f" diff={avg[3]:.4f}"
                print(f"[step {step}/{max_steps}] loss={avg[0]:.4f} "
                      f"(video {avg[1]:.4f}, audio {avg[2]:.4f}){ref} grad_norm={grad_norm:.4f} "
                      f"t_mean={avg[4]:.3f} lr={lr_at(step - 1, lr_schedule):.2e} "
                      f"{dt:.1f}s/step peak={peak:.1f}GiB {fs.phase_summary()}", flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(avg[0].item(), 6),
                                        "loss_video": round(avg[1].item(), 6),
                                        "loss_audio": round(avg[2].item(), 6),
                                        "loss_diffusion": round(avg[3].item(), 6),

                                        # Comparable ACROSS objectives, unlike loss_diffusion:
                                        # the sampling variance cancels in the difference.
                                        "loss_diffusion_teacher": round(avg[6].item(), 6),
                                        "excess_vs_teacher": round((avg[3] - avg[6]).item(), 6),

                                        # The pure teacher term even when lambda > 0, so a
                                        # mixed arm's curve stays comparable to a lambda=0 one.
                                        "loss_teacher_only": round(avg[5].item(), 6),
                                        "grad_norm": round(float(grad_norm), 6),
                                        "timestep_mean": round(avg[4].item(), 4),
                                        "lr": lr_at(step - 1, lr_schedule),
                                        "s_per_step": round(dt, 2), "peak_gib": round(peak, 2),
                                        **fs.phase_fields()}) + "\n")
            t_log = time.time()
            fs.phase_reset()
            reset_peak_memory_stats(device)
            periodic = step % cfg.checkpoint.save_every == 0 or step == max_steps
            if periodic or step in early_saves:
                save_weights(step)
            if periodic:
                save_state(step, epoch, in_epoch)
            if step >= max_steps:
                break
        epoch += 1
        in_epoch = 0
        resumable_sampler.skip = 0

    dist.barrier()
    if rank == 0:
        print(f"stage {stage.upper()} done", flush=True)
        run_lock.release(output_dir)
    dist.destroy_process_group()


def main():
    cfg = load_config(StageBConfig)
    run(cfg)


if __name__ == "__main__":
    main()
