"""Stage A2: end-to-end alignment of the linear branch to the full-attention teacher,
architecture INHERITED from the A1 checkpoint.

    torchrun --standalone --nproc_per_node=8 src/training/train_stage_a2.py \\
        --config configs/training/stage_a2_c1_vdn_anchor.yaml \\
        [initialization.checkpoint=... optimizer.lr=1e-4 ...]

The schema has NO model section: an architecture override dies as an unknown key.
The FSDP2/HSDP machinery -- guards, fp32 alpha islands, split-block checkpointing,
activation offload, fingerprinted saves, name-keyed train states, sharded resume -- is
src/training/fsdp_stage.py, shared with Stage B. What is
specific to A2 lives here: the loss, the trainable set (every transform parameter,
optionally minus the softmax gate), the big/small optimizer split and the metrics row.
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import load_config
from src.config.stage_a2 import StageA2Config
from src.models.factory import build_model, load_model_weights
from src.models.hybrid_transform import (is_transform_parameter, set_layout,
                                         set_softmax_backend)
from src.training import fsdp_stage as fs
from src.training.fsdp_stage import _PHASE, phase
from src.training.t2va_batch import pack_noisy_batch
from src.utils import run_lock
from src.utils.distributed import broadcast_and_verify_init
from src.utils.lr_classes import lr_class_map
from src.utils.lr_schedule import lr_at
from src.utils.device import (distributed_backend, local_device,
                              max_memory_allocated, reset_peak_memory_stats,
                              set_device, set_memory_fraction, synchronize)

STAGE = "a2"
WEIGHTS_PREFIX = "hybrid_step"


def compute_loss(model, sample, device, noise_generator, sigma_generator,
                 audio_loss_weight, offload_activations):
    with phase("data"):
        batch = pack_noisy_batch(sample, device, noise_generator, sigma_generator)
        set_layout(model, batch["layout"])
    _PHASE["seq_len"] = float(batch["layout"].seq_len)   # last micro-batch's length

    # Teacher target only — the reason this stage exists. The diffusion target
    # x0 - noise is a ONE-SAMPLE draw whose irreducible variance is most of the ~0.5
    # loss floor; the teacher's output approximates E[v | x_t], the quantity that loss
    # estimates, so the target is variance-free by comparison — and it keeps A1 -> A2
    # in one objective family (per-layer alignment -> end-to-end alignment).
    with phase("teacher_fwd"):
        target_v, target_a = fs.teacher_velocity(model, batch["inputs"])
    with phase("student_fwd"):
        velocity_v, velocity_a = fs.student_forward(model, batch["inputs"],
                                                    offload_activations)
    pred_v, pred_a = velocity_v[0].float(), velocity_a[0].float()

    loss_video = torch.nn.functional.mse_loss(pred_v, target_v)
    loss_audio = torch.nn.functional.mse_loss(pred_a, target_a)
    loss = loss_video + audio_loss_weight * loss_audio

    # The data-target MSE, detached, as the shared metric: teacher-target losses live on
    # their own scale, so without this number an A2 curve cannot be compared to any
    # diffusion-objective run's.
    with torch.no_grad():
        loss_ref = (torch.nn.functional.mse_loss(pred_v, batch["target_video_rows"])
                    + audio_loss_weight
                    * torch.nn.functional.mse_loss(pred_a, batch["target_audio_rows"]))

    # Zeroed THROUGH the graph so FSDP's reduction still sees every trainable parameter.
    if not torch.isfinite(loss):
        print(f"non-finite loss at t_v={batch['t_v']:.4f}; zeroing this step", flush=True)
        loss = (velocity_v.float().sum() + velocity_a.float().sum()) * 0.0
        loss_video = loss_audio = loss_ref = loss.detach()

    return loss, loss_video.detach(), loss_audio.detach(), loss_ref, batch["t_v"]


def profile_step(prof_step, step, micro, grad_accum, rank):
    """H3_PROF_STEP=N: on rank 0, wrap the micro-batch that completes step N in
    torch.profiler (CUDA kernels only, aggregated — no trace file) and print the kernel
    table + a bucket summary. Only rank 0 profiles; the others just wait at the
    collectives, so the step's wall time is inflated but attribution holds."""
    if not (prof_step and torch.cuda.is_available() and rank == 0 and step + 1 == prof_step
            and (micro + 1) % grad_accum == 0):
        return None
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    prof.__enter__()
    return prof


def report_profile(prof, prof_step):
    synchronize("cuda")
    prof.__exit__(None, None, None)
    evs = prof.key_averages()
    print(f"=== H3_PROF step {prof_step}: top kernels (rank 0) ===", flush=True)
    print(evs.table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
    buckets = {}
    for ev in evs:
        n = ev.key.lower()
        for pat, b in (("conv_depthwise", "conv_depthwise3d"), ("nccl", "nccl"),
                       ("gemm", "matmul"), ("cutlass", "matmul"), ("flash", "attention"),
                       ("fmha", "attention"), ("memcpy", "memcpy"),
                       ("elementwise", "elementwise"), ("reduce_kernel", "reduce")):
            if pat in n:
                key = b
                break
        else:
            key = "other"
        t_us = getattr(ev, "self_device_time_total",
                       getattr(ev, "self_cuda_time_total", 0))
        buckets[key] = buckets.get(key, 0.0) + t_us
    print("=== H3_PROF bucket summary (self CUDA time) ===", flush=True)
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v/1e6:8.1f} s", flush=True)
    print(f"  {'TOTAL':18s} {sum(buckets.values())/1e6:8.1f} s", flush=True)


def main():
    cfg = load_config(StageA2Config)
    sched = SimpleNamespace(lr=cfg.optimizer.lr, min_lr=cfg.optimizer.min_lr,
                            warmup_steps=cfg.optimizer.warmup_steps,
                            max_steps=cfg.training.max_steps)
    out_dir = cfg.checkpoint.output_dir
    grad_accum = cfg.training.gradient_accumulation_steps
    max_steps = cfg.training.max_steps

    fs.refuse_if_guarded(out_dir, os.path.join(out_dir, f"{WEIGHTS_PREFIX}{max_steps:06d}.pt"),
                         cfg.training.ignore_stopped)

    device = local_device()
    set_device(device)
    dist.init_process_group(distributed_backend(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    set_memory_fraction(cfg.distributed.memory_fraction, device)
    torch.manual_seed(cfg.training.seed)   # same-seed init; FSDP2 does not broadcast

    # -------- build seam: the A1 artifact decides EVERYTHING structural --------
    art = fs.load_seed_artifact(cfg.initialization.checkpoint)
    spec_dict = art.model_spec
    if rank == 0:
        fs.print_architecture(spec_dict, "A1 architecture")
    model = build_model(spec_dict, device="cpu", base_source=cfg.initialization.base_source)
    set_softmax_backend(model, cfg.runtime.kernels.softmax_backend, device=device)
    loaded = load_model_weights(model, art.weights)
    del art
    model.requires_grad_(False)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"DiT built from spec + {loaded} A1 tensors loaded: {n / 1e9:.1f}B params",
              flush=True)

    # Trainables: the transform's parameters, fp32 masters (compute is bf16). The
    # softmax gate can be held at its A1 value; it is still SAVED by name.
    trainable_params, n_branch = [], 0
    for name, param in model.named_parameters():
        if not is_transform_parameter(name):
            continue
        param.data = param.data.to(torch.float32)
        if cfg.training.freeze_softmax_gate and ".softmax_gate." in name:
            continue
        param.requires_grad_(True)
        trainable_params.append(param)
        n_branch += param.numel()
    if not trainable_params:
        raise RuntimeError("no branch parameters found -- was the transform applied?")
    if rank == 0:
        print(f"trainables: branch {n_branch / 1e6:.1f}M params (no LoRA)", flush=True)

    resume = fs.Resume()
    if cfg.training.auto_resume:
        resume = fs.find_resume(out_dir, WEIGHTS_PREFIX, spec_dict, sched, model, rank)

    broadcast_and_verify_init(model, trainable_params, device)
    if rank == 0:
        print("init broadcast + cross-rank verification passed", flush=True)

    model = fs.shard_model(
        model, world, cfg.distributed.shard_size, device, rank,
        activation_checkpointing=cfg.distributed.activation_checkpointing)

    # lr classes AFTER sharding (identity-keyed).
    classes = lr_class_map(model)
    groups = fs.build_param_groups(model, [
        ("big", cfg.optimizer.big_lr_scale,
         lambda n, p: classes.get(id(p), "big") == "big", None),
        ("small", cfg.optimizer.small_lr_scale,
         lambda n, p: classes.get(id(p), "big") == "small", cfg.optimizer.small_eps),
    ], cfg.optimizer.lr)
    if rank == 0:
        print(fs.describe_groups(groups), flush=True)   # before AdamW fills defaults in
    optimizer = torch.optim.AdamW(groups, lr=cfg.optimizer.lr,
                                  weight_decay=cfg.optimizer.weight_decay)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    weight_names = [n for n, _ in model.named_parameters() if is_transform_parameter(n)]

    dataset, sampler, skip_sampler, loader = fs.make_loader(
        cfg.data.index_file, cfg.data.num_workers, world, rank, cfg.training.seed)
    if rank == 0:
        print(f"dataset: {len(dataset)} clips, {len(loader)} steps/epoch/rank, "
              f"global batch {world * grad_accum}", flush=True)
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
    noise_generator, sigma_generator = fs.make_generators(cfg.training.seed, rank, device)
    epoch = fs.restore_optimizer_and_rng(resume, model, optimizer, noise_generator,
                                         sigma_generator, skip_sampler, world, rank)

    def save_weights(step):
        fs.save_weights_artifact(model, out_dir, f"{WEIGHTS_PREFIX}{step:06d}.pt", STAGE,
                                 step, rank, spec_dict,
                                 select=lambda n, p: is_transform_parameter(n))

    def save_state(step, epoch, in_epoch):
        fs.save_train_state(model, optimizer, weight_names, trainable_names, STAGE, step,
                            epoch, in_epoch, noise_generator, sigma_generator, cfg,
                            spec_dict, out_dir, rank, world, cfg.checkpoint.keep_states)

    model.train()
    step, in_epoch, micro = resume.start_step, resume.in_epoch, 0
    early_saves = set(cfg.checkpoint.early_saves)
    if 0 in early_saves and step == 0:
        save_weights(0)
    acc = torch.zeros(5, device=device)  # loss, video, audio, loss_diffusion, t_v
    prof_step = int(os.environ.get("H3_PROF_STEP", "0") or "0")
    t_log = time.time()
    while step < max_steps:
        sampler.set_epoch(epoch)
        for sample in loader:
            in_epoch += 1
            prof = profile_step(prof_step, step, micro, grad_accum, rank)
            loss, loss_video, loss_audio, loss_ref, t_v = compute_loss(
                model, sample, device, noise_generator, sigma_generator,
                cfg.training.audio_loss_weight, cfg.distributed.offload_activations)
            with phase("backward"):
                (loss / grad_accum).backward()
            if prof is not None:
                report_profile(prof, prof_step)
            acc += torch.stack([loss.detach(), loss_video, loss_audio, loss_ref,
                                torch.tensor(t_v, device=device)])
            micro += 1
            if micro % grad_accum != 0:
                continue

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, sched) * group["lr_scale"]
            grad_norm = fs.clip_gradients(trainable, cfg.optimizer.grad_clip)
            with phase("optim"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1

            avg = acc / grad_accum
            acc = torch.zeros_like(acc)
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            if rank == 0:
                dt = time.time() - t_log
                peak = max_memory_allocated(device) / 2**30
                print(f"[step {step}/{max_steps}] loss={avg[0]:.4f} "
                      f"(video {avg[1]:.4f}, audio {avg[2]:.4f}) diff={avg[3]:.4f} "
                      f"grad_norm={grad_norm:.4f} "
                      f"t_mean={avg[4]:.3f} lr={lr_at(step - 1, sched):.2e} "
                      f"{dt:.1f}s/step peak={peak:.1f}GiB {fs.phase_summary()}", flush=True)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(avg[0].item(), 6),
                                        "loss_video": round(avg[1].item(), 6),
                                        "loss_audio": round(avg[2].item(), 6),
                                        "loss_diffusion": round(avg[3].item(), 6),
                                        "grad_norm": round(float(grad_norm), 6),
                                        "timestep_mean": round(avg[4].item(), 4),
                                        "lr": lr_at(step - 1, sched),
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
        skip_sampler.skip = 0

    dist.barrier()
    if rank == 0:
        print("stage A2 done", flush=True)
        run_lock.release(out_dir)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
