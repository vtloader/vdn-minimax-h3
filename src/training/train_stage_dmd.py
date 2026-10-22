"""Stage-DMD: DMD2 (no GAN) for the few-step Turbo LoRA on the frozen Stage-B VDN.

    torchrun --standalone --nproc_per_node=8 src/training/train_stage_dmd.py \\
        --config configs/training/stage_dmd_vdn.yaml [training.max_steps=500 ...]

Three roles on ONE FSDP model (src/training/dmd.py): the generator (VDN + the frozen
Stage-B `default` LoRA + trainable `turbo`, initialised from an external few-step
LoRA), the real score
(`dmd.real_score`: the released dense model, or the frozen VDN itself) and the fake
score (the real score's architecture + trainable `fake` LoRA, B=0 at init). Only `turbo`
and `fake` train. Weights artifacts ship transform + `default` + `turbo` -- what
inference merges -- and the train_state additionally carries `fake` and the fake update
counter, so a resume lands mid-recipe rather than re-warming the fake.

The loop has DMD2's shape. One sub-iteration = one prompt per rank (no latents are
loaded; the generator samples), and every sub-iteration does

  rollout     generator, k Euler steps from pure noise, no_grad, k uniform on the grid
              and identical on every rank (it sets the number of collectives)
  sample      grid step k -> x0_g (with a graph only on a generator turn)
  fake x 1    flow-matching regression of `fake` on the detached x0_g at (u', eps')

and on a GENERATOR turn -- the first of every `dmd.fake_updates_per_step` sub-iterations
after the warm-up -- additionally re-noises x0_g at u ~ U[u_min, u_max] (paired 12u/3u),
runs the real + fake score forwards (no_grad), takes the DMD loss and steps `turbo`.
So fake:generator = R:1 on R DIFFERENT samples, one of them shared with the generator,
exactly as in DMD2. The first `dmd.fake_warmup_steps` sub-iterations are fake-only
(generator frozen, samples still drawn). One AdamW, two groups, two cosine schedules:
a parameter without a gradient in a given backward is skipped by both AdamW and FSDP2,
so a generator step moves `turbo` only and a fake step moves `fake` only.

Evaluation is renders, nothing else: the DM loss is not monotone. Every sub-iteration
logs its turn, the DM loss (generator turns), the DMD normaliser, the real-vs-fake x0
gap, the fake regression loss and the phase timers to metrics.jsonl.
"""
import copy
import gc
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames, audio_latent_num_frames, video_latent_num_frames)

from src.config import load_config
from src.config.stage_dmd import VDN_ADAPTER_NAME, StageDMDConfig, validate_stage_dmd
from src.models.factory import build_model, inject_adapters, load_model_weights
from src.models.hybrid_transform import (is_transform_parameter, set_layout,
                                         set_softmax_backend)
from src.models.model_spec import ModelSpec
from src.training import dmd
from src.training import fsdp_stage as fs
from src.training.fsdp_stage import _PHASE, phase
from src.training.t2va_batch import PackedPrompt, initial_noise, sample_paired_timesteps
from src.training.turbo_adapter import (copy_adapter_state, extra_shard_module_paths,
                                        fake_adapter_config, inject_lora_adapter,
                                        is_named_adapter_parameter, load_external_adapter)
from src.utils import run_lock
from src.utils.distributed import broadcast_and_verify_init
from src.utils.lr_schedule import lr_at
from src.utils.device import (distributed_backend, empty_cache, local_device,
                              max_memory_allocated, reset_peak_memory_stats,
                              set_device, set_memory_fraction)

STAGE = "d"
WEIGHTS_PREFIX = "hybrid_turbo_lora_step"
VDN_ADAPTER = VDN_ADAPTER_NAME
LATENT_CHANNELS = 24

# metrics vector, all-reduced (AVG) across ranks before logging
_METRIC_KEYS = ("loss", "loss_video", "loss_audio", "normaliser_video", "normaliser_audio",
                "gap_video", "gap_audio", "fake_loss", "fake_loss_video", "fake_loss_audio",
                "grad_norm", "fake_grad_norm", "rollout_index", "u")


def is_lora(name):
    return "lora_" in name


def _seed_steps(cfg, artifact):
    candidates = [
        ("initialization_checkpoint.step", getattr(artifact, "step", None)),
        ("metadata.step", artifact.metadata.get("step")),
        ("checkpoint", cfg.initialization.checkpoint),
        ("metadata.converted_from", artifact.metadata.get("converted_from")),
    ]
    found = []
    for source, candidate in candidates:
        if isinstance(candidate, int):
            found.append((source, candidate))
        elif isinstance(candidate, str):
            match = re.search(r"step(\d+)\.pt$", candidate)
            if match:
                found.append((source, int(match.group(1))))
    return found


def validate_seed(cfg, artifact):
    """The seed is the one Stage-B step the config names, carrying exactly its adapter."""
    found = _seed_steps(cfg, artifact)
    expected = cfg.initialization.source_step
    if not found or any(step != expected for _, step in found):
        actual = ", ".join(f"{source}={step}" for source, step in found) or "unknown"
        raise RuntimeError(f"Stage-DMD requires a step-{expected} VDN seed; detected {actual}")
    adapters = artifact.model_spec.get("adapters") or []
    if len(adapters) != 1:
        raise RuntimeError("Stage-DMD requires exactly the one inherited Stage-B VDN adapter")
    if adapters[0].get("config", {}).get("name", VDN_ADAPTER) != VDN_ADAPTER:
        raise RuntimeError(f"the seed's adapter must be the {VDN_ADAPTER!r} Stage-B LoRA")


def resolve_stage_dmd_model_spec(source_model_spec, turbo_config):
    """Stamp the trainable Turbo adapter; the fake score never enters an artifact."""
    resolved = copy.deepcopy(source_model_spec)
    resolved.setdefault("adapters", []).append(
        {"type": "lora", "version": 1, "config": copy.deepcopy(turbo_config)})
    ModelSpec.from_dict(resolved)
    return resolved


def generation_geometry(generation):
    """(video latent shape, num_audio_latents) of the generator's output, from the
    same helpers the render uses, so a training sample IS an eval render's shape."""
    frames = align_num_frames(generation.num_frames, 17, 5)
    num_latent_frames = video_latent_num_frames(frames, 17, 5)
    num_audio_latents = audio_latent_num_frames(frames)
    video_shape = (LATENT_CHANNELS, num_latent_frames,
                   generation.latent_height, generation.latent_width)
    return video_shape, int(num_audio_latents)


def zero_loss_like(*tensors):
    """A finite zero that still reaches every trainable through the graph."""
    return sum(t.float().sum() for t in tensors) * 0.0


def run(cfg):
    gen_schedule = SimpleNamespace(lr=cfg.optimizer.lr, min_lr=cfg.optimizer.min_lr,
                                   warmup_steps=cfg.optimizer.warmup_steps,
                                   max_steps=cfg.training.max_steps)
    updates_per_step = cfg.dmd.fake_updates_per_step
    warmup_target = cfg.dmd.fake_warmup_steps
    fake_schedule = SimpleNamespace(
        lr=cfg.optimizer.fake_lr, min_lr=cfg.optimizer.fake_min_lr,
        warmup_steps=cfg.optimizer.fake_warmup_steps,
        max_steps=warmup_target + updates_per_step * cfg.training.max_steps)
    output_dir = cfg.checkpoint.output_dir
    max_steps = cfg.training.max_steps
    offload = cfg.distributed.offload_activations
    audio_weight = cfg.training.audio_loss_weight
    turbo_name, fake_name = cfg.turbo.adapter_name, cfg.dmd.fake_adapter_name
    u_min, u_max = cfg.dmd.u_min, cfg.dmd.u_max
    video_shift, audio_shift = cfg.turbo.video_shift, cfg.turbo.audio_shift
    final_path = os.path.join(output_dir, f"{WEIGHTS_PREFIX}{max_steps:06d}.pt")
    fs.refuse_if_guarded(output_dir, final_path, cfg.training.ignore_stopped)

    device = local_device()
    set_device(device)
    dist.init_process_group(distributed_backend(device))
    rank, world = dist.get_rank(), dist.get_world_size()
    set_memory_fraction(cfg.distributed.memory_fraction, device)
    torch.manual_seed(cfg.training.seed)   # same-seed init on every rank (fake's lora_A)

    seed_artifact = fs.load_seed_artifact(cfg.initialization.checkpoint)
    validate_seed(cfg, seed_artifact)
    source_model_spec = seed_artifact.model_spec
    if rank == 0:
        fs.print_architecture(source_model_spec, label="frozen VDN architecture")

    model = build_model(source_model_spec, device="cpu",
                        base_source=cfg.initialization.base_source)
    set_softmax_backend(model, cfg.runtime.kernels.softmax_backend, device=device)
    transform_weights = {name: tensor for name, tensor in seed_artifact.weights.items()
                         if not is_lora(name)}
    loaded_transforms = load_model_weights(model, transform_weights)
    model.requires_grad_(False)

    # The frozen Stage-B adapter first, under its `default` name.
    model = inject_adapters(model, source_model_spec)
    vdn_lora_weights = {name: tensor for name, tensor in seed_artifact.weights.items()
                        if is_lora(name)}
    loaded_vdn = load_model_weights(model, vdn_lora_weights)

    # `turbo`: the external adapter's bytes (mixed ranks). `fake`: the same target set
    # at one uniform rank, PEFT's zero lora_B, so fake == real score at step 0.
    turbo_state, turbo_config = load_external_adapter(model, cfg.turbo.checkpoint, turbo_name)
    turbo_config["family"] = cfg.turbo.family
    extra_shards = extra_shard_module_paths(turbo_config)
    model = inject_lora_adapter(model, turbo_config, turbo_name)
    copied_turbo = copy_adapter_state(model, turbo_state, turbo_name)
    fake_config = fake_adapter_config(turbo_config, cfg.dmd.fake_rank, fake_name)
    model = inject_lora_adapter(model, fake_config, fake_name)
    resolved_model_spec = resolve_stage_dmd_model_spec(source_model_spec, turbo_config)

    del seed_artifact, transform_weights, vdn_lora_weights, turbo_state
    model.requires_grad_(False)
    unsharded_trainables = []
    counts = {turbo_name: 0, fake_name: 0}
    fake_b_max = 0.0
    for name, param in model.named_parameters():
        for adapter in (turbo_name, fake_name):
            if is_named_adapter_parameter(name, adapter):
                param.data = param.data.to(torch.float32)
                param.requires_grad_(True)
                unsharded_trainables.append(param)
                counts[adapter] += param.numel()
                if adapter == fake_name and f".lora_B.{fake_name}." in name:
                    fake_b_max = max(fake_b_max, float(param.detach().abs().max()))
    trainable_names = [name for name, param in model.named_parameters()
                       if param.requires_grad]
    wrong = [name for name in trainable_names
             if not (is_named_adapter_parameter(name, turbo_name)
                     or is_named_adapter_parameter(name, fake_name))]
    if wrong or not counts[turbo_name] or not counts[fake_name]:
        raise RuntimeError(f"Stage-DMD trainable boundary violated: {wrong[:4] or counts}")
    if fake_b_max != 0.0:
        raise RuntimeError("fake adapter lora_B is not zero at init; fake != real at step 0")
    roles = dmd.Roles(model, cfg.dmd.real_score, VDN_ADAPTER, turbo_name, fake_name)
    roles.set(dmd.GENERATOR)
    if rank == 0:
        total = sum(param.numel() for param in model.parameters())
        print(f"DiT built: {total / 1e9:.1f}B params; restored {loaded_transforms} "
              f"transform + {loaded_vdn} frozen VDN LoRA tensors", flush=True)
        print(f"turbo ({cfg.turbo.family}): {copied_turbo} tensors, "
              f"{counts[turbo_name] / 1e6:.1f}M trainable; fake: rank {cfg.dmd.fake_rank}, "
              f"{counts[fake_name] / 1e6:.1f}M trainable, lora_B == 0", flush=True)
        print(f"roles (real_score={cfg.dmd.real_score}): "
              + "; ".join(f"{role}: teacher_mode={roles.teacher_mode(role)} "
                          f"adapters={roles.adapters(role)}"
                          for role in (dmd.GENERATOR, dmd.REAL, dmd.FAKE)), flush=True)
        print(f"schedule: {cfg.turbo.num_steps} NFE, shifts {video_shift:g}/{audio_shift:g}; "
              f"DM u in [{u_min:g}, {u_max:g}]; fake:generator {updates_per_step}:1 on "
              f"{updates_per_step} fresh samples per generator step; fake warm-up "
              f"{warmup_target} sub-iterations", flush=True)

    resume = fs.Resume()
    if cfg.training.auto_resume:
        resume = fs.find_resume(
            output_dir, WEIGHTS_PREFIX, resolved_model_spec, gen_schedule, model, rank)
    fake_step = iteration = 0
    if resume.full_state is not None:
        fake_step = int(resume.full_state.metadata.get("fake_step", 0))
        iteration = int(resume.full_state.metadata.get("iteration", 0))
    elif resume.start_step and rank == 0:
        print("weights-only resume: the fake score restarts from B=0 and re-warms",
              flush=True)

    broadcast_and_verify_init(model, unsharded_trainables, device)
    if rank == 0:
        print("turbo/fake init broadcast + cross-rank verification passed", flush=True)
    del unsharded_trainables, param
    gc.collect()

    model = fs.shard_model(
        model, world, cfg.distributed.shard_size, device, rank,
        activation_checkpointing=cfg.distributed.activation_checkpointing,
        extra_module_paths=extra_shards)
    gc.collect()
    empty_cache(device)
    roles.set(dmd.GENERATOR)

    optimizer_groups = fs.build_param_groups(model, [
        (turbo_name, 1.0, lambda name, param: is_named_adapter_parameter(name, turbo_name),
         None),
        (fake_name, 1.0, lambda name, param: is_named_adapter_parameter(name, fake_name),
         None),
    ], cfg.optimizer.lr)
    if len(optimizer_groups) != 2:
        raise RuntimeError("expected exactly the turbo and fake optimizer groups")
    if rank == 0:
        print(fs.describe_groups(optimizer_groups), flush=True)
        print(f"saved-tensor CPU offload: {offload}", flush=True)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=cfg.optimizer.lr,
                                  weight_decay=cfg.optimizer.weight_decay)
    group_of = {group["name"]: group for group in optimizer.param_groups}
    turbo_parameters = list(group_of[turbo_name]["params"])
    fake_parameters = list(group_of[fake_name]["params"])
    trainable_names = [name for name, param in model.named_parameters()
                       if param.requires_grad]
    shipped_names = [
        name for name, _ in model.named_parameters()
        if (is_transform_parameter(name)
            or is_named_adapter_parameter(name, VDN_ADAPTER)
            or is_named_adapter_parameter(name, turbo_name))
    ]
    shipped_name_set = set(shipped_names)

    video_shape, num_audio_latents = generation_geometry(cfg.generation)
    dataset, distributed_sampler, resumable_sampler, data_loader = fs.make_loader(
        cfg.data.index_file, cfg.data.num_workers, world, rank, cfg.training.seed,
        text_only=True)
    if rank == 0:
        print(f"dataset: {len(dataset)} prompts (text only), {len(data_loader)} "
              f"sub-iterations/epoch/rank ({updates_per_step} per generator step); "
              f"generating {video_shape} video latents + "
              f"{num_audio_latents} audio latents (dataset clips are {dataset.video_shape})",
              flush=True)
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
    noise_generator, timestep_generator = fs.make_generators(cfg.training.seed, rank, device)
    epoch = fs.restore_optimizer_and_rng(
        resume, model, optimizer, noise_generator, timestep_generator,
        resumable_sampler, world, rank)
    schedule = dmd.FewStepSchedule(cfg.turbo.num_steps, video_shift, audio_shift)

    provenance = {
        "objective": "dmd2_no_gan",
        "real_score": cfg.dmd.real_score,
        "teacher": ("dense_released" if cfg.dmd.real_score == "dense"
                    else f"vdn_stage_b_step{cfg.initialization.source_step}"),
        "student": f"stage_b_vdn+{cfg.turbo.family}_trainable",
        "turbo_num_steps": cfg.turbo.num_steps,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "fake_rank": cfg.dmd.fake_rank,
        "fake_updates_per_step": updates_per_step,
        "fake_warmup_steps": warmup_target,
    }

    def save_weights(step):
        fs.save_weights_artifact(
            model, output_dir, f"{WEIGHTS_PREFIX}{step:06d}.pt", STAGE, step, rank,
            resolved_model_spec,
            select=lambda name, param: name in shipped_name_set,
            metadata=provenance)

    def save_state(step, current_epoch, in_epoch, fake_step, iteration):
        fs.save_train_state(
            model, optimizer, trainable_names, trainable_names, STAGE, step,
            current_epoch, in_epoch, noise_generator, timestep_generator, cfg,
            resolved_model_spec, output_dir, rank, world, cfg.checkpoint.keep_states,
            extra_metadata={"fake_step": fake_step, "iteration": iteration})

    def set_learning_rates(step, fake_step):
        group_of[turbo_name]["lr"] = lr_at(step, gen_schedule)
        group_of[fake_name]["lr"] = lr_at(fake_step, fake_schedule)
        return group_of[turbo_name]["lr"], group_of[fake_name]["lr"]

    def draw_noise_like(x0_v, x0_a):
        return (torch.randn(x0_v.shape, generator=noise_generator, device=device,
                            dtype=torch.float32),
                torch.randn(x0_a.shape, generator=noise_generator, device=device,
                            dtype=torch.float32))

    def fake_update(packed, x0_v, x0_a, step, fake_step):
        """One flow-matching update of `fake` on the detached generator sample."""
        t_v, t_a, _ = sample_paired_timesteps(timestep_generator, u_min, u_max,
                                              video_shift, audio_shift)
        noise_v, noise_a = draw_noise_like(x0_v, x0_a)
        x_v, x_a = dmd.noised(x0_v, t_v, noise_v), dmd.noised(x0_a, t_a, noise_a)
        roles.set(dmd.FAKE)
        velocity_v, velocity_a = fs.student_forward(
            model, packed.inputs(x_v, x_a, t_v, t_a), offload)
        loss_v = dmd.fake_regression_loss(velocity_v[0].float(), x0_v, noise_v)
        loss_a = dmd.fake_regression_loss(velocity_a[0].float(), x0_a, noise_a)
        loss = loss_v + audio_weight * loss_a
        if not torch.isfinite(loss):
            print(f"non-finite fake loss at fake update {fake_step}; zeroing", flush=True)
            loss = zero_loss_like(velocity_v, velocity_a)
            loss_v = loss_a = loss.detach()
        dmd.require_role(roles, dmd.FAKE, "the fake backward")
        loss.backward()
        grad_norm = fs.clip_gradients(fake_parameters, cfg.optimizer.fake_grad_clip)
        set_learning_rates(step, fake_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach(), loss_v.detach(), loss_a.detach(), float(grad_norm)

    model.train()
    step = resume.start_step
    in_epoch = resume.in_epoch
    early_saves = set(cfg.checkpoint.early_saves)
    if 0 in early_saves and step == 0 and fake_step == 0:
        save_weights(0)   # renders identically to Stage-B + the unmodified external adapter
    if rank == 0 and fake_step < warmup_target:
        print(f"fake warm-up: {warmup_target - fake_step} fake-only sub-iterations to go "
              f"(generator frozen, samples still drawn)", flush=True)

    t_log = time.time()
    while step < max_steps:
        distributed_sampler.set_epoch(epoch)
        for sample in data_loader:
            in_epoch += 1
            turn = dmd.turn(fake_step, warmup_target, updates_per_step)
            generator_turn = turn == dmd.GENERATOR_TURN
            fs.phase_reset()
            reset_peak_memory_stats(device)
            metrics = {key: 0.0 for key in _METRIC_KEYS}

            with phase("data"):
                packed = PackedPrompt(sample["prompt_embeds"], sample["text_token_tags"],
                                      video_shape, num_audio_latents, device)
                set_layout(model, packed.layout)
            _PHASE["seq_len"] = float(packed.seq_len)

            # k is rank-shared (it sets the number of forwards, i.e. of collectives);
            # the noise and the DM/fake timesteps stay per-rank streams.
            index = dmd.shared_rollout_index(cfg.training.seed, iteration, schedule.num_steps)
            iteration += 1
            video_rows, audio_rows = initial_noise(video_shape, num_audio_latents,
                                                   noise_generator, device)
            with phase("rollout"):
                video_rows, audio_rows = dmd.rollout(
                    model, roles, schedule, packed, video_rows, audio_rows, index)
                empty_cache(device)
            metrics["rollout_index"] = float(index)

            with phase("generator_fwd"):
                x0_v, x0_a = dmd.generator_x0(
                    model, roles, schedule, packed, video_rows, audio_rows, index,
                    offload, build_graph=generator_turn)
                if not generator_turn:
                    empty_cache(device)

            if generator_turn:
                t_v, t_a, u = sample_paired_timesteps(timestep_generator, u_min, u_max,
                                                      video_shift, audio_shift)
                noise_v, noise_a = draw_noise_like(x0_v, x0_a)
                x_v = dmd.noised(x0_v.detach(), t_v, noise_v)
                x_a = dmd.noised(x0_a.detach(), t_a, noise_a)
                with phase("score_fwd"):
                    x0_rv, x0_ra = dmd.score_x0(model, roles, dmd.REAL, packed,
                                                x_v, x_a, t_v, t_a)
                    x0_fv, x0_fa = dmd.score_x0(model, roles, dmd.FAKE, packed,
                                                x_v, x_a, t_v, t_a)
                del x_v, x_a, noise_v, noise_a
                loss_v, norm_v, gap_v = dmd.distribution_matching_loss(x0_v, x0_rv, x0_fv)
                loss_a, norm_a, gap_a = dmd.distribution_matching_loss(x0_a, x0_ra, x0_fa)
                del x0_rv, x0_ra, x0_fv, x0_fa
                loss = loss_v + audio_weight * loss_a
                if not torch.isfinite(loss):
                    print(f"non-finite DM loss at step {step + 1} (grid {index}, u {u:.3f}); "
                          f"zeroing", flush=True)
                    loss = zero_loss_like(x0_v, x0_a)
                    loss_v = loss_a = loss.detach()
                dmd.require_role(roles, dmd.GENERATOR, "the generator backward")
                with phase("backward"):
                    loss.backward()
                grad_norm = fs.clip_gradients(turbo_parameters, cfg.optimizer.grad_clip)
                set_learning_rates(step, fake_step)
                with phase("optim"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                metrics.update(loss=float(loss.detach()), loss_video=float(loss_v.detach()),
                               loss_audio=float(loss_a.detach()), normaliser_video=float(norm_v),
                               normaliser_audio=float(norm_a), gap_video=float(gap_v),
                               gap_audio=float(gap_a), grad_norm=float(grad_norm), u=u)
                del loss, loss_v, loss_a

            # The fake sees this sub-iteration's sample (shared with the generator on a
            # generator turn), at a fresh noise level and noise.
            x0_v, x0_a = x0_v.detach(), x0_a.detach()
            with phase("fake_update"):
                fake_loss, fake_loss_v, fake_loss_a, fake_norm = fake_update(
                    packed, x0_v, x0_a, step, fake_step)
                fake_step += 1
            metrics.update(fake_loss=float(fake_loss), fake_loss_video=float(fake_loss_v),
                           fake_loss_audio=float(fake_loss_a), fake_grad_norm=fake_norm)
            del packed, x0_v, x0_a, video_rows, audio_rows

            avg = torch.tensor([metrics[key] for key in _METRIC_KEYS], device=device)
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            lr_gen, lr_fake = set_learning_rates(step, fake_step)
            if rank == 0:
                elapsed = time.time() - t_log
                peak = max_memory_allocated(device) / 2**30
                row = {key: round(float(value), 6) for key, value in zip(_METRIC_KEYS, avg)}
                row.update(step=step, fake_step=fake_step, iteration=iteration, turn=turn,
                           lr=lr_gen, lr_fake=lr_fake, s_per_step=round(elapsed, 2),
                           peak_gib=round(peak, 2), **fs.phase_fields())
                if turn == dmd.WARMUP_TURN:
                    tag = f"[warm-up fake {fake_step}/{warmup_target}]"
                else:
                    tag = f"[step {step}/{max_steps} {'G' if generator_turn else 'F'} fake {fake_step}]"
                dm = (f"dm={row['loss']:.4f} (v {row['loss_video']:.4f} a {row['loss_audio']:.4f}) "
                      f"norm={row['normaliser_video']:.4f} gap={row['gap_video']:.4f} "
                      if generator_turn else "")
                print(f"{tag} {dm}fake={row['fake_loss']:.4f} k={row['rollout_index']:.0f} "
                      f"u={row['u']:.3f} gn={row['grad_norm']:.4f}/{row['fake_grad_norm']:.4f} "
                      f"lr={lr_gen:.2e}/{lr_fake:.2e} {elapsed:.1f}s peak={peak:.1f}GiB "
                      f"{fs.phase_summary()}", flush=True)
                with open(metrics_path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            t_log = time.time()

            if turn == dmd.WARMUP_TURN:
                if fake_step >= warmup_target:
                    if rank == 0:
                        print("fake warm-up complete; generator updates begin", flush=True)
                    save_state(step, epoch, in_epoch, fake_step, iteration)
                continue
            if generator_turn:
                periodic = step % cfg.checkpoint.save_every == 0 or step == max_steps
                if periodic or step in early_saves:
                    save_weights(step)
                if periodic:
                    save_state(step, epoch, in_epoch, fake_step, iteration)
                if step >= max_steps:
                    break
        epoch += 1
        in_epoch = 0
        resumable_sampler.skip = 0

    dist.barrier()
    if rank == 0:
        print("stage D done", flush=True)
        run_lock.release(output_dir)
    dist.destroy_process_group()


def main():
    cfg = load_config(StageDMDConfig, extra_validators=[validate_stage_dmd])
    run(cfg)


if __name__ == "__main__":
    main()
