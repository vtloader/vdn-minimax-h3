"""The render shared by every inference entrypoint: generate_latents (the packed layout and
the denoising loop) and decode_and_save with its atomic .partial.mp4 write.
"""
import os
import time
from types import SimpleNamespace

import torch

from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3Ref2VAPrepareLayoutStep, patchify_video_latents)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS as AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG as AUDIO_TAG,
    MINIMAX_H3_FPS as FPS, MINIMAX_H3_VIDEO_TAG as VIDEO_TAG, align_num_frames,
    audio_latent_num_frames, video_latent_num_frames)
from diffusers.utils.export_utils import encode_video

from src.inference.utils.prompt_cache import is_reference_request
from src.inference.geometry import EXPORT_FPS, export_frame_count, validate_canvas
from src.models.sequence_layout import layout_from_indices
from src.models.hybrid_transform import iter_hybrids, set_layout
from src.utils.device import autocast, generator, synchronize

# The production canvas, 768x1344, through the 16x spatial VAE.
LATENT_H, LATENT_W = 48, 84
# Mirrors of MiniMaxH3ModularPipeline.pixel_mean / pixel_std / keyframe_noise_aug: those
# are properties of an assembled pipeline, which the inlined blocks below never build.
# 0.999 is the t the fl2va keyframes are held at, just short of clean.
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KEYFRAME_NOISE_AUG = 0.999


@torch.no_grad()
def generate_latents(transformer, prompt_embeds, text_token_tags, num_frames, num_steps, seed, device,
                     video_shift=12.0, audio_shift=3.0, runtime=None, step_seconds=None,
                     conditions=None, progress_callback=None, width=None, height=None):
    """The sampler. `runtime` (Ulysses) adds a barrier before and after the loop so every
    rank enters and leaves together; `step_seconds`, if a list, receives the
    device-synchronised wall time of every NFE (for the timing log and the record).

    `conditions` = (keyframe_anchors, condition_latents) from load_prompt turns the
    request into i2va / fl2va: the layout reserves one frame of conditioning rows per
    keyframe between the text and the audio (`[text | keyframes | audio | video]`), the
    anchors are noised to t = 0.999 and pinned there, and only the generated rows are
    ever stepped -- the diffusers fl2va blocks, inlined. On a hybrid model the
    conditioning rows fall outside the layout's video span, i.e. they are attended like
    text and audio: dense in both directions by the window softmax and absent from the
    linear scan.

    Anchors that are all "ref" (encode_keyframes.py --refs) are image references instead:
    the diffusers ref2va layout, `[text | one block per reference | audio | video]`, each
    block on its own geometry and its own rotary slot ahead of the generated rows. The
    noise, the pinning and the hybrid treatment are the keyframes', and the transformer
    is whatever was loaded, the t2va/fl2va one, not MiniMax-H3's transformer_ref."""
    num_frames = align_num_frames(num_frames, 17, 5)
    num_latent_frames = video_latent_num_frames(num_frames, 17, 5)
    if num_latent_frames <= 0:
        raise ValueError(
            f"num_frames={num_frames} produces no video latent frames; "
            "use at least 22 frames (the valid 17*n+5 geometry)"
        )
    num_audio_latents = audio_latent_num_frames(num_frames)
    anchors, condition_latents = conditions if conditions else ((), [])
    if width is not None or height is not None:
        validate_canvas(width if width is not None else LATENT_W * 16,
                        height if height is not None else LATENT_H * 16)
    latent_h = height // 16 if height is not None else LATENT_H
    latent_w = width // 16 if width is not None else LATENT_W
    patch = tuple(transformer.config.patch_size)                    # (1, 2, 2)
    channels = transformer.config.in_channels                       # 24
    frame_h, frame_w = latent_h // patch[1], latent_w // patch[2]
    if not is_reference_request(anchors):
        for condition in condition_latents:
            if tuple(condition.shape[-2:]) != (latent_h, latent_w):
                raise ValueError("keyframe cache canvas differs from the render canvas; "
                                 f"re-encode with --width {latent_w * 16} --height {latent_h * 16}")

    if is_reference_request(anchors):
        # The layout step reads a reference's modality only (`kind`, `has_audio`); the
        # geometry comes from the latents.
        references = [SimpleNamespace(kind="image", has_audio=False) for _ in anchors]
        layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            text_token_tags, references, condition_latents, [], num_latent_frames, latent_h,
            latent_w, num_audio_latents, patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG,
        )
    else:
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags, num_latent_frames, latent_h, latent_w, num_audio_latents,
            patch, AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, keyframe_anchors=anchors,
        )
    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, _ = layout
    position_ids, token_tags = position_ids.to(device), token_tags.to(device)
    video_indices, audio_indices, text_indices = (
        video_indices.to(device), audio_indices.to(device), text_indices.to(device),
    )

    # A hybrid-converted transformer needs the packed layout before every forward; the
    # layout is the same at every denoising step, so set it once per generation.
    if next(iter_hybrids(transformer), None) is not None:
        # frame_size and text_indices unconditionally: carrying them is free, and only
        # their consumers are gated (short_conv / text_state).
        set_layout(transformer, layout_from_indices(
            video_indices[num_condition_rows:], num_latent_frames, frame_h * frame_w,
            seq_len=position_ids.shape[0], frame_size=(frame_h, frame_w),
            text_indices=text_indices,
        ))

    # The scheduler counts sigma grid points, the terminal 0 included: one more than the
    # num_steps model evaluations.
    scheduler = MiniMaxH3Scheduler(shift=video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=audio_shift)
    scheduler.set_timesteps(num_steps + 1, device=device)
    audio_scheduler.set_timesteps(num_steps + 1, device=device)

    noise_generator = generator(device, seed)
    # The conditioning noise is drawn first, one draw per keyframe or reference, before
    # the generated rows' noise (MiniMaxH3PrepareConditionLatentsStep's order).
    condition_rows = []
    for condition in condition_latents:
        noise = torch.randn(condition.shape, generator=noise_generator, device=device,
                            dtype=torch.float32)
        noised = scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, noise)
        condition_rows.append(patchify_video_latents(noised, patch))
    latents = torch.randn((1, channels, num_latent_frames, latent_h, latent_w),
                          generator=noise_generator, device=device, dtype=torch.float32)
    video_rows = patchify_video_latents(latents, patch)
    if condition_rows:
        video_rows = torch.cat(condition_rows + [video_rows])
        if video_rows.shape[0] != video_indices.numel():
            raise ValueError(f"{video_rows.shape[0]} video rows (with conditioning) != "
                             f"{video_indices.numel()} layout rows")
    audio_rows = torch.randn((num_audio_latents * AUDIO_CHANNELS, 32),
                             generator=noise_generator, device=device, dtype=torch.float32)

    if runtime is not None:
        runtime.barrier()
        synchronize(device)

    seq_len = position_ids.shape[0]
    for step, (t, audio_t) in enumerate(zip(scheduler.timesteps, audio_scheduler.timesteps), 1):
        step_started = time.perf_counter()
        row_timesteps = torch.full((seq_len,), float(t), dtype=torch.float32, device=device)
        if num_condition_rows:
            row_timesteps[video_indices[:num_condition_rows]] = max(float(t), KEYFRAME_NOISE_AUG)
        row_timesteps[audio_indices] = float(audio_t)
        timestep, timestep_indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        noise_pred, audio_noise_pred = transformer(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds[None],
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=False,
        )
        # only the generated rows step; the anchors ride through unchanged
        video_rows[num_condition_rows:] = scheduler.step(
            noise_pred[0, num_condition_rows:].float(), t, video_rows[num_condition_rows:],
            return_dict=False)[0]
        audio_rows = audio_scheduler.step(audio_noise_pred[0].float(), audio_t, audio_rows, return_dict=False)[0]

        if step_seconds is not None or progress_callback is not None:
            synchronize(device)
            elapsed = time.perf_counter() - step_started
            if step_seconds is not None:
                step_seconds.append(elapsed)
            if progress_callback is not None:
                progress_callback(step, len(scheduler.timesteps), elapsed)

    if runtime is not None:
        synchronize(device)
        runtime.barrier()

    # Unpatchify (the AfterDenoise step's reshape) and unpack the channel-major audio rows.
    video_rows = video_rows[num_condition_rows:]
    rows = video_rows.reshape(-1, num_latent_frames, frame_h, frame_w, channels, *patch)
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    latents = rows.reshape(-1, channels, num_latent_frames, latent_h, latent_w).contiguous()
    audio_latents = audio_rows.reshape(AUDIO_CHANNELS, num_audio_latents, 32).permute(0, 2, 1).contiguous()
    return latents, audio_latents


@torch.no_grad()
def decode_and_save(latents, audio_latents, vae, audio_vae, out_path: str, device: str, fps: int = FPS):
    if fps not in EXPORT_FPS:
        raise ValueError(f"fps must be one of {EXPORT_FPS}")
    vae.to(device)
    audio_vae.to(device)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    with autocast(device, dtype=torch.float16):
        video = vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    audio = audio_vae.decode(audio_latents * audio_latents_std + audio_latents_mean, return_dict=False)[0]
    audio = audio.float().permute(1, 0, 2)[0]  # (2, num_samples)

    # The decoders are only needed for this call. They were loaded on CPU, so put them
    # back there before the encode: decode is the last consumer of HBM either way, but
    # this keeps the peak bounded on a device that also holds resident transformer
    # blocks (or a later re-render in the same process).
    vae.to("cpu")
    audio_vae.to("cpu")

    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()  # (F, H, W, 3)
    if fps != FPS:
        frames = resample_video_frames(frames, fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Encode to a sibling temp name and rename into place, so a render that is killed
    # mid-encode never leaves a truncated clip under the final name (anything that
    # treats "the mp4 exists" as "done" would otherwise pick it up). os.replace is
    # atomic within a directory, so the final name only ever appears complete.
    # The suffix goes BEFORE the extension: PyAV infers the container format from the
    # output filename, and "clip.mp4.partial" makes av.open raise "Could not determine
    # output format". Same directory either way, which is what os.replace needs.
    stem, ext = os.path.splitext(out_path)
    partial_path = f"{stem}.partial{ext}"
    encode_video(frames, fps=fps, output_path=partial_path,
                 audio=audio.cpu(), audio_sample_rate=audio_vae.config.sampling_rate)
    os.replace(partial_path, out_path)
    print(f"wrote {out_path}", flush=True)


def resample_video_frames(frames, fps):
    """Nearest-frame export resampling, preserving duration and both endpoint images.

    Motion and audio are generated on the model's fixed 24 FPS timeline. This only
    drops/duplicates decoded frames; it does not synthesize intermediate motion.
    """
    count = export_frame_count(len(frames), fps)
    indices = torch.linspace(0, len(frames) - 1, count).round().long()
    return frames.index_select(0, indices)
