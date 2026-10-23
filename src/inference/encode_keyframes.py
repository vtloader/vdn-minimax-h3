"""Encode a keyframe request -- prompt + first and/or last keyframe -- and cache it as
one prompt .pt that infer.py renders like any t2va cache, plus the keyframe
conditioning. Which keyframes you pass is the mode: `--first` alone is i2va, `--last`
alone is l2va, both is fl2va.

    python src/inference/encode_keyframes.py --prompt "..." \\
        --first first.png --last last.png --out prompts/mine.pt

Replicates the diffusers `fl2va` presentation exactly (MiniMaxH3ResizeStep,
MiniMaxH3FL2VATextEncoderStep, MiniMaxH3KeyframeVaeEncoderStep): the first keyframe is
stretched onto the 768-short-edge canvas of its own aspect ratio, the last is
cover-cropped onto it; the presentation is `"<Picture i>: " + vision block` per keyframe
(vision rows tagged as VIDEO) then the prompt verbatim; the Qwen3-VL hidden state after
layer 50 conditions the transformer; each keyframe is VAE-encoded under the fixed seed 42
posterior sample, rounded to fp16, and normalised. The .pt carries:

    prompt, prompt_embeds (L, 5120) bf16, text_token_tags (L,), keyframe_anchors
    ["first", "last"], condition_latents [(1, 24, 1, H/16, W/16) fp32 ...], height, width

`--refs a.png b.png ...` instead of keyframes is a ref2va-like request: the images are
presented and packed the way the diffusers `ref2va` blocks do it (MiniMaxH3Ref2VASetupStep,
MiniMaxH3Ref2VATextEncoderStep, MiniMaxH3Ref2VAReferenceEncoderStep), each on a short edge
of its own (`--ref_size`, default 768; the released Ref2VA recipe's is 2048, four times the
tokens), labelled `"<Picture i>: "` + vision block and VAE-encoded at that size, and the
render is by the same t2va/fl2va transformer, not MiniMax-H3's `transformer_ref`. Every
reference is anchored "ref", which render.py keys the ref2va layout off; the .pt also
carries `reference_size`.

    python src/inference/encode_keyframes.py --prompt "..." \\
        --refs ref-1.png ref-2.png --out prompts/mine.pt

Add `--device npu:1 --offload` (or `--device cuda:0 --offload`) to keep the Qwen
encoder on CPU between leaf-module calls. The VAE still runs on the selected device.
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VATextEncoderStep, encode_vae_condition)
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size

from src.inference.render import PIXEL_MEAN, PIXEL_STD
from src.inference.geometry import validate_canvas
from src.inference.utils.prompt_cache import REFERENCE_ANCHOR, conditioning_mode
from src.paths import H3_BASE, resolve_weights, upstream_snapshot
from src.utils.device import device_of, empty_cache, set_device

TEXT_ENCODER_LAYER = 50
VIDEO_TAG, TEXT_TAG = 0, 1
CANVAS_MULTIPLE, CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS = 32, 768, 768 * 1344
MAX_ASPECT = 4
KEYFRAME_ENCODE_SEED = 42
REFERENCE_SHORT_EDGE = 768

# (anchors) -> a substring of the mode's instruction, the instruction. MiniMax-H3 expects
# the prompt to open with the line for its mode (VIDEO_PROMPT_WRITING_GUIDE_base_en.md);
# i2va's is fixed, the other two name the final shot and the duration.
KEYFRAME_INSTRUCTIONS = {
    ("first",): ("is fully referenced",
                 "For the target video, at 0.00 seconds into the target video, "
                 "<Picture 1> (from [Shot 1]) is fully referenced."),
    ("last",): ("How the reference pictures align",
                "How the reference pictures align with the target video \u2014 <Picture 1> "
                "(from [Shot N]) aligns with the S.SS-second mark of the target video."),
    ("first", "last"): ("How the reference pictures align",
                        "How the reference pictures align with the target video \u2014 Picture 1 "
                        "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                        "Picture 2 (from Shot N) aligns with the S.SS-second mark of the "
                        "target video."),
}


def put_on_canvas(keyframes, width=None, height=None):
    """MiniMaxH3ResizeStep: the canvas follows the first keyframe's aspect ratio; the
    first keyframe is stretched onto it, every follower cover-cropped (the released
    model's rounding, not VaeImageProcessor's). Returns (keyframes, height, width)."""
    if width is not None or height is not None:
        validate_canvas(width, height)
        return [ImageOps.fit(image.convert("RGB"), (width, height),
                             method=Image.Resampling.LANCZOS) for image in keyframes], height, width
    height, width = resolve_canvas_size(*keyframes[0].size, CANVAS_MULTIPLE,
                                        CANVAS_SHORT_EDGE, CANVAS_MAX_PIXELS)
    prepared = []
    for index, keyframe in enumerate(keyframes):
        keyframe = keyframe.convert("RGB")
        if keyframe.size == (width, height):
            prepared.append(keyframe)
        elif index == 0:
            prepared.append(keyframe.resize((width, height), Image.Resampling.LANCZOS))
        else:
            scale = max(width / keyframe.size[0], height / keyframe.size[1])
            resized_size = (max(width, round(keyframe.size[0] * scale)),
                            max(height, round(keyframe.size[1] * scale)))
            left = max(0, (resized_size[0] - width) // 2)
            top = max(0, (resized_size[1] - height) // 2)
            resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
            prepared.append(resized.crop((left, top, left + width, top + height)))
    return prepared, height, width


def normalize_references(images, short_edge=REFERENCE_SHORT_EDGE):
    """MiniMaxH3Ref2VASetupStep's rule for an image reference, with `short_edge` in place
    of its 2048: each image on a short edge of `short_edge` at its own aspect ratio
    (upscaled if smaller, no area cap), both sides rounded to the canvas multiple,
    LANCZOS. A reference never binds the target canvas."""
    prepared = []
    for image in images:
        image = image.convert("RGB")
        width, height = image.size
        if width > MAX_ASPECT * height or height > MAX_ASPECT * width:
            raise ValueError(f"a reference image must be within 1:{MAX_ASPECT} and "
                             f"{MAX_ASPECT}:1, got {width}x{height}")
        scale = short_edge / min(width, height)
        target = tuple(max(CANVAS_MULTIPLE, round(side * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                       for side in (width, height))
        prepared.append(image if image.size == target
                        else image.resize(target, Image.Resampling.LANCZOS))
    return prepared


def build_presentation(processor, prompt, keyframes):
    """MiniMaxH3FL2VATextEncoderStep's tokenisation: label + vision block per keyframe,
    then the prompt verbatim; returns token ids, per-row tags, vision inputs."""
    tokenizer = processor.tokenizer
    vision = processor.image_processor(images=keyframes, return_tensors="pt")
    image_grid_thw = vision["image_grid_thw"]
    vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": image_grid_thw}
    token_ids, token_tags = [], []
    merge_size = processor.image_processor.merge_size ** 2
    for index in range(len(keyframes)):
        num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
        label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        vision_ids = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                      + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                      + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        token_ids += label_ids + vision_ids
        token_tags += [TEXT_TAG] * len(label_ids) + [VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    return token_ids, token_tags, vision_inputs


@torch.no_grad()
def qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, device):
    """diffusers' get_qwen3vl_prompt_embeds, inlined: hidden state after decoder layer 50
    of `text_encoder.model` with the vision tensors and Qwen-internal mm_token_type_ids."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]),
                                     dtype=torch.long, device=device)
    vision_kwargs = {name: (value.to(device, text_encoder.dtype) if name.startswith("pixel_")
                            else value.to(device)) for name, value in vision_inputs.items()}
    outputs = text_encoder.model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids, use_cache=False, output_hidden_states=True,
        **vision_kwargs)
    return outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()


def check_instruction(prompt, anchors):
    """The mode these keyframes select. Warns on stderr when the prompt does not open
    with that mode's instruction line -- never refuses: the check is a substring, and a
    prompt is free text."""
    mode = conditioning_mode(anchors)
    marker, template = KEYFRAME_INSTRUCTIONS[tuple(anchors)]
    opening = next((line for line in prompt.splitlines() if line.strip()), "")
    if marker.lower() not in opening.lower():
        print(f"warning: {mode} keyframes, but the prompt does not open with {mode}'s "
              f"instruction line. MiniMax-H3 expects it as the first line:\n"
              f"    {template}\nEncoding the prompt as given.", file=sys.stderr, flush=True)
    return mode


def encode_mixed_references(args, device, model_root):
    """Encode a ref2va request containing images, videos and/or audio clips."""
    num_frames = int(args.num_frames)
    if num_frames < 22 or (num_frames - 5) % 17:
        raise ValueError(f"ref2va num_frames must be 17*n+5, got {num_frames}")
    if args.width is not None:
        width, height = args.width, args.height
    else:
        height, width = resolve_canvas_size(16, 9, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE,
                                            CANVAS_MAX_PIXELS)

    references, files, kinds = [], [], []
    for path in args.refs or []:
        image = normalize_references([Image.open(path)], args.ref_size)[0]
        references.append(MiniMaxH3ImageReference(image=image))
        files.append(path); kinds.append("image")
    for path in args.video_refs or []:
        raw = MiniMaxH3VideoReference.from_file(path)
        frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            raw.frames, raw.fps, num_frames, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE,
            CANVAS_MAX_PIXELS, 24.0)
        references.append(MiniMaxH3VideoReference(
            frames=frames, fps=24.0, audio=raw.audio, sample_rate=raw.sample_rate))
        files.append(path); kinds.append("video")
    for path in args.audio_refs or []:
        audio = MiniMaxH3AudioReference.from_file(path)
        references.append(audio)
        files.append(path); kinds.append("audio")

    print(f"ref2va: canvas {width}x{height}; references {kinds}", flush=True)
    print(f"Loading processor from {model_root}", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(model_root, subfolder="processor")
    text_step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = (
        text_step._gather_vision_features(processor, references, 24.0))
    token_ids, token_tags = text_step._build_presentation(
        processor.tokenizer, args.prompt, references, image_counts, video_counts,
        video_timestamps, text_tag=TEXT_TAG, video_tag=VIDEO_TAG)

    print(f"Loading text encoder ({'CPU leaf offload' if args.offload else device})", flush=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_root, subfolder="text_encoder", dtype=torch.bfloat16)
    text_encoder.eval().requires_grad_(False)
    if args.offload:
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(
            text_encoder.model, onload_device=device, offload_device="cpu",
            offload_type="leaf_level", use_stream=False)
    else:
        text_encoder.to(device)
    print(f"Encoding {len(token_ids)} text/vision tokens on {device}", flush=True)
    prompt_embeds = qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, device)
    del text_encoder
    gc.collect()
    empty_cache(device)

    print(f"Loading VAE from {args.vae_root} on {device}", flush=True)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        resolve_weights(args.vae_root), subfolder="vae").to(device).eval()
    needs_audio = any(reference.has_audio for reference in references)
    audio_vae = None
    if needs_audio:
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            resolve_weights(args.vae_root), subfolder="audio_vae").to(device).eval()
        audio_sr = int(audio_vae.config.sampling_rate)
        audio_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
        audio_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)
        max_duration = num_frames / 24.0

    condition_latents, audio_condition_latents = [], []
    with torch.no_grad():
        for reference in references:
            if reference.kind == "image":
                pixels = torch.from_numpy(np.array(reference.image)).to(device)
                pixels = pixels.permute(2, 0, 1)[None, :, None]
                condition_latents.append(encode_vae_condition(
                    vae, pixels, PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED))
            elif reference.kind == "video":
                frames = reference.frames[:num_frames]
                chunk = int(vae.config.clip_length)
                latents_chunk = int(vae.tokens_chunk_size)
                count = max(1, (frames.shape[0] - latents_chunk) // chunk) * chunk + latents_chunk
                frames = frames[:count]
                pixels = torch.from_numpy(frames.copy()).to(device).permute(3, 0, 1, 2)[None]
                condition_latents.append(encode_vae_condition(
                    vae, pixels, PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED))

            if not reference.has_audio:
                continue
            waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
                reference.audio, reference.sample_rate or audio_sr, audio_sr, max_duration)
            posterior = audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
            latents = posterior.mode().float().cpu().transpose(1, 2)
            normalized = (latents - audio_mean) / audio_std
            audio_condition_latents.append(
                normalized.reshape(-1, audio_vae.config.latent_channels))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    return {
        "prompt": args.prompt,
        "prompt_embeds": prompt_embeds,
        "text_token_tags": torch.tensor(token_tags, dtype=torch.long),
        "keyframe_anchors": [REFERENCE_ANCHOR] * len(references),
        "keyframe_files": files,
        "reference_kinds": kinds,
        "reference_has_audio": [reference.has_audio for reference in references],
        "condition_latents": condition_latents,
        "audio_condition_latents": audio_condition_latents,
        "height": height,
        "width": width,
        "reference_size": args.ref_size,
        "num_frames": num_frames,
    }


def main():
    p = argparse.ArgumentParser(description="Cache Qwen3-VL and VAE conditioning for keyframes or references.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--first", type=str, help="keyframe the video starts from (image path)")
    p.add_argument("--last", type=str, help="keyframe the video ends on (image path)")
    p.add_argument("--refs", type=str, nargs="+", metavar="IMAGE",
                   help="reference images in the order the prompt numbers them "
                        "(<Picture 1>, <Picture 2>, ...); a ref2va-like request")
    p.add_argument("--ref_size", type=int, default=REFERENCE_SHORT_EDGE,
                    help="short edge every image reference is put on")
    p.add_argument("--video_refs", type=str, nargs="+", metavar="VIDEO",
                   help="reference videos in packed order, after images")
    p.add_argument("--audio_refs", type=str, nargs="+", metavar="AUDIO",
                   help="reference audio clips in packed order, after videos")
    p.add_argument("--num_frames", type=int, default=345,
                   help="generated frame count; reference video/audio are truncated to this duration")
    p.add_argument("--width", type=int, default=None,
                   help="explicit output canvas width, paired with --height; multiples of 32")
    p.add_argument("--height", type=int, default=None,
                   help="explicit output canvas height; keyframes are cover-cropped to this canvas")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--model_root", type=str, default=None,
                   help="a MiniMax-H3 snapshot with processor/ and text_encoder/; "
                        "default: downloaded from the Hub on first use")
    p.add_argument("--vae_root", type=str, default=H3_BASE, help="root holding vae/")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="encoding device, e.g. cuda:0 or npu:1")
    p.add_argument("--offload", action="store_true",
                   help="keep Qwen encoder weights on CPU and move leaf modules to --device "
                        "only while encoding (slower; no stream prefetch); VAE stays on --device")
    args = p.parse_args()
    if args.width is not None or args.height is not None:
        try:
            validate_canvas(args.width, args.height)
        except ValueError as exc:
            p.error(str(exc))
    mixed = bool(args.video_refs or args.audio_refs)
    if mixed and (args.first or args.last):
        p.error("--video_refs/--audio_refs are ref2va-only and cannot be combined with keyframes")
    if mixed and not (args.refs or args.video_refs):
        p.error("ref2va needs at least one image or video reference; audio cannot stand alone")
    if not mixed and bool(args.refs) == bool(args.first or args.last):
        p.error("pass --first and/or --last, or --refs (a request with neither is t2va: "
                 "use encode_prompt.py)")
    device = device_of(args.device)
    set_device(device)
    if args.model_root is None:
        print("Resolving processor and text encoder snapshot", flush=True)
        args.model_root = upstream_snapshot("processor", "text_encoder")

    if mixed:
        cache = encode_mixed_references(args, device, args.model_root)
        torch.save(cache, args.out)
        print(f"wrote {args.out}: {cache['text_token_tags'].numel()} tokens, "
              f"{len(cache['condition_latents'])} visual and "
              f"{len(cache['audio_condition_latents'])} audio reference(s)", flush=True)
        return

    if args.refs:
        pairs = [(REFERENCE_ANCHOR, path) for path in args.refs]
        anchors = [anchor for anchor, _ in pairs]
        mode = conditioning_mode(anchors)
        keyframes = normalize_references([Image.open(path) for _, path in pairs], args.ref_size)
        # MiniMaxH3Ref2VASetupStep's canvas: 16:9, whatever the references' shapes.
        if args.width is not None:
            width, height = args.width, args.height
        else:
            height, width = resolve_canvas_size(16, 9, CANVAS_MULTIPLE, CANVAS_SHORT_EDGE,
                                                CANVAS_MAX_PIXELS)
        print(f"{mode}: canvas {width}x{height}; references "
              f"{[f'{k.size[0]}x{k.size[1]}' for k in keyframes]}", flush=True)
    else:
        pairs = [(anchor, path) for anchor, path in (("first", args.first), ("last", args.last)) if path]
        anchors = [anchor for anchor, _ in pairs]
        mode = check_instruction(args.prompt, anchors)
        canvas_options = {} if args.width is None else {"width": args.width, "height": args.height}
        keyframes, height, width = put_on_canvas([Image.open(path) for _, path in pairs], **canvas_options)
        print(f"{mode}: canvas {width}x{height}; keyframes {anchors}", flush=True)
    print(f"Loading processor from {args.model_root}", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(args.model_root, subfolder="processor")
    token_ids, token_tags, vision_inputs = build_presentation(processor, args.prompt, keyframes)
    print(f"Loading text encoder ({'CPU leaf offload' if args.offload else device})", flush=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_root, subfolder="text_encoder", dtype=torch.bfloat16)
    text_encoder.eval().requires_grad_(False)
    if args.offload:
        from diffusers.hooks import apply_group_offloading

        # Encoding calls the inner model directly, bypassing the outer LM forward.
        apply_group_offloading(
            text_encoder.model, onload_device=device, offload_device="cpu",
            offload_type="leaf_level", use_stream=False,
        )
    else:
        text_encoder.to(device)
    print(f"Encoding {len(token_ids)} text/vision tokens on {device}", flush=True)
    prompt_embeds = qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, device)
    print("Releasing text encoder before VAE encoding", flush=True)
    del text_encoder
    gc.collect()  # Group-offload hooks hold cyclic references to their modules.
    empty_cache(device)

    print(f"Loading VAE from {args.vae_root} on {device}", flush=True)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(resolve_weights(args.vae_root), subfolder="vae").to(device)
    vae.eval()
    print(f"VAE-encoding {len(keyframes)} image(s) on {device}", flush=True)
    with torch.no_grad():
        condition_latents = [
            encode_vae_condition(vae, torch.from_numpy(np.array(k)).to(device).permute(2, 0, 1)[None, :, None],
                                 PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED)
            for k in keyframes]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "prompt": args.prompt,
        "prompt_embeds": prompt_embeds,                                   # (L, 5120) bf16
        "text_token_tags": torch.tensor(token_tags, dtype=torch.long),   # vision rows tagged VIDEO
        "keyframe_anchors": anchors,
        "keyframe_files": [path for _, path in pairs],
        "condition_latents": condition_latents,
        "height": height, "width": width,
        **({"reference_size": args.ref_size} if args.refs else {}),
    }, args.out)
    print(f"wrote {args.out}: {len(token_ids)} tokens ({token_tags.count(VIDEO_TAG)} vision rows), "
          f"{len(keyframes)} keyframes {[tuple(c.shape) for c in condition_latents]}", flush=True)


if __name__ == "__main__":
    main()
