"""One model assembly for every inference entrypoint.

    model = build_inference_model(cfg, device)          # infer.py
    model = build_inference_model(cfg, device, load_decoders=runtime.is_main,
                                  log=runtime.is_main)  # infer_ulysses.py

Order of operations, the same for both: read the artifact's spec -> load the base ->
apply the transform -> load the branch weights -> fold the artifact's LoRAs -> fold the
external LoRAs -> eval/no-grad on device -> behaviour -> the inference kernel set
(set_inference_mode) -> window-softmax backend -> ablation -> fp8. Then `render_record`
says what ACTUALLY ran (fp8 layer count, the flex latch, the resolved backend), so a
silent downgrade is never silent in the JSON next to the mp4.

Every knob is a config field (src/config/inference.py); nothing here reads an
environment variable. `checkpoint: null` is the released DENSE model: no transform, no
branch, and the hybrid-only overlays are skipped.
"""
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diffusers import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
                       MiniMaxH3Transformer3DModel)

from src.checkpoints import checkpoint_head_sha256, load_checkpoint
from src.config import resolved_dict
from src.inference.utils.lora import load_external_lora, merge_lora_state
from src.inference.utils.npu_pinned_offload import apply_npu_pinned_offload
from src.paths import H3_BASE, resolve_weights
from src.models.factory import load_model_weights
from src.models.hybrid_transform import (apply_hybrid_attention_transform, iter_hybrids,
                                         set_inference_mode, set_softmax_backend)
from src.models.ops.fp8_linear import convert_linear_to_fp8
from src.models.softmax_attention.flex_attention import _FLEX_CACHE
from src.utils.device import device_of, empty_cache, is_cuda, synchronize

# Semantic knobs an ablation may touch, and where they live. Anything else is refused.
ABLATABLE = ("radius", "chunk", "anchor_frames", "bridge", "enable_text_state",
             "linear_attention_enabled", "enable_softmax_gate")


@dataclass
class InferenceModel:
    transformer: Any
    vae: Any                                  # None on ranks that do not decode
    audio_vae: Any
    artifact: Any                             # CheckpointArtifact, or None for dense
    is_hybrid: bool
    merged_lora_pairs: int = 0
    external_loras: List[Dict[str, Any]] = field(default_factory=list)
    ablated: List[str] = field(default_factory=list)
    fp8_linears: int = 0
    softmax_backend: Optional[str] = None     # resolved (auto -> flex|decomposed)
    npu_resident_blocks: int = 0              # actual count left out of the offload hooks
    npu_pinned_offload: Optional[Dict[str, Any]] = None
    total_blocks: int = 0
    resident_blocks_released: bool = False    # True once evicted before decode
    resident_release_seconds: Optional[float] = None


def apply_ablation(model, overrides):
    for key in overrides:
        if key not in ABLATABLE:
            raise ValueError(f"{key!r} is not an ablatable knob ({ABLATABLE})")

    for attn in iter_hybrids(model):
        for key, value in overrides.items():
            setattr(attn, key, value)

    return sorted(overrides)


def flex_latch_state():
    return {"compiled_variants": [k for k in ("infer", "train") if k in _FLEX_CACHE],
            "flash_available": _FLEX_CACHE.get("flash_available")}


def _load_decoders(vae_source: str = None):
    vae_source = resolve_weights(vae_source or H3_BASE)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(vae_source, subfolder="vae")
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(vae_source, subfolder="audio_vae")
    vae.eval()
    audio_vae.eval()
    return vae, audio_vae


def load_decoders(vae_source: str = None):
    """Load both decoders on the CPU from `vae_source` (default: the release copy)."""
    return _load_decoders(vae_source)


def load_models(model_root: str, device: str, vae_source: str = None,
                load_decoders: bool = True, offload_transformer: bool = False):
    """`model_root` holds the transformer; both decoders come from `vae_source`, default
    the release copy (H3_BASE: transformer/ vae/ audio_vae/, which the dense
    `checkpoint=null` render loads from too). Ranks that never decode pass
    `load_decoders=False` and get (transformer, None, None).

    The decoders stay on the CPU: `decode_and_save` brings them to the GPU when
    denoising is over. Their 10 GB next to the bf16 transformer, which is what the card
    holds until the fp8 conversion, is what an 80 GB card does not have."""
    vae_source = resolve_weights(vae_source or H3_BASE)
    model_root = resolve_weights(model_root)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        model_root, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if not offload_transformer:
        transformer = transformer.to(device)
    transformer.eval().requires_grad_(False)

    if not load_decoders:
        return transformer, None, None

    vae, audio_vae = _load_decoders(vae_source)
    return transformer, vae, audio_vae


def apply_transformer_offload(transformer, device, resident_blocks: int = 0) -> int:
    """Install Diffusers group-offload hooks after every custom forward is in place and
    leave the first `resident_blocks` DiT blocks resident on `device`.

    With resident_blocks=0 this is exactly the block-level offload the NPU path has
    always used. Otherwise the hook installation sees a temporary ModuleList holding
    only the blocks that still stream from CPU. That keeps Diffusers' own grouping and
    onload/offload hooks untouched, and it keeps the resident blocks out of the
    top-level "unmatched" group, so no hook ever moves them. The full list is restored
    before the caller touches the model again.

    The last block must stay offloaded: the full static transformer is 65.7 GiB and
    does not fit a 64 GiB NPU, so full residency is refused instead of left to OOM.
    """
    from diffusers.hooks import apply_group_offloading

    blocks = transformer.transformer_blocks
    total = len(blocks)
    resident_blocks = int(resident_blocks)
    if resident_blocks < 0 or resident_blocks > total:
        raise ValueError(f"npu_resident_blocks={resident_blocks} outside 0..{total}")
    if resident_blocks == total:
        raise ValueError(
            f"npu_resident_blocks={total} would keep the whole 65.7 GiB transformer "
            "resident; keep at least one block offloaded on a 64 GiB NPU"
        )

    if resident_blocks:
        offloaded = torch.nn.ModuleList(list(blocks[resident_blocks:]))
        transformer.transformer_blocks = offloaded
        try:
            apply_group_offloading(
                transformer,
                onload_device=device,
                offload_device="cpu",
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=False,
            )
        finally:
            transformer.transformer_blocks = blocks

        for block in blocks[:resident_blocks]:
            block.to(device)
        empty_cache(device)
        return resident_blocks

    apply_group_offloading(
        transformer,
        onload_device=device,
        offload_device="cpu",
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )
    return 0


def install_inference_offload(transformer, device, cfg, *, say=print):
    """Install either the NPU pinned-host path or the pageable group offload.

    Returns ``(resident_blocks, pinned_state)``.  The two paths are mutually exclusive
    at config validation time; the pinned path leaves every block offloaded.
    """
    if getattr(cfg.kernels, "npu_pinned_offload", False):
        state = apply_npu_pinned_offload(transformer, device)
        available = state.get("host_available_after_pin_bytes")
        available_gib = "unknown" if available is None else f"{available / (1024 ** 3):.1f}"
        say(f"npu pinned offload: {state['tensors_pinned']} tensors, "
            f"{state['model_gib']:.2f} GiB pinned in {state['pin_seconds']:.1f}s, "
            f"{state['pinned_cache_bytes'] / (1024 ** 3):.1f} GiB pinned cache, "
            f"{available_gib} GiB host available", flush=True)
        return 0, state

    resident = apply_transformer_offload(
        transformer, device, resident_blocks=cfg.kernels.npu_resident_blocks)
    if resident:
        total = len(transformer.transformer_blocks)
        say(f"npu residency: {resident}/{total} blocks resident on {device}, "
            f"{total - resident} streamed from CPU", flush=True)
    return resident, None


def release_resident_blocks(model: "InferenceModel", device) -> bool:
    """Move the transformer's resident blocks back to CPU before the VAE onloads.

    Only the first `model.npu_resident_blocks` blocks are resident; every other block
    already sits on CPU between forwards. Safe to call when nothing is resident (the
    Ulysses ranks without a decoder, or npu_resident_blocks=0) and idempotent.
    """
    resident = model.npu_resident_blocks
    if not resident or model.resident_blocks_released:
        return False

    started = time.perf_counter()
    synchronize(device)
    for block in model.transformer.transformer_blocks[:resident]:
        block.to("cpu")
    empty_cache(device)
    model.resident_blocks_released = True
    model.resident_release_seconds = time.perf_counter() - started
    return True


def build_inference_model(cfg, device, *, load_decoders: bool = True,
                          log: bool = True, install_offload: bool = True,
                          softmax_device=None) -> InferenceModel:
    say = print if log else (lambda *a, **k: None)
    if cfg.precision.fp8.enabled and not is_cuda(device):
        raise ValueError("precision.fp8.enabled requires CUDA; use bf16 for NPU inference")

    art = None
    if cfg.checkpoint:
        art = load_checkpoint(resolve_weights(cfg.checkpoint))
        if art.metadata.get("truncated_blocks"):
            raise RuntimeError(f"{cfg.checkpoint} is a truncated smoke-test artifact")

    base_source = cfg.base_source or (
        art.model_spec["base"]["source"] if art else H3_BASE)
    transformer, vae, audio_vae = load_models(
        base_source, device, vae_source=cfg.vae_source, load_decoders=load_decoders,
        offload_transformer=cfg.render.offload_dit)

    merged = 0
    if art:
        apply_hybrid_attention_transform(
            transformer, art.model_spec["transforms"][0]["config"])

        branch_weights = {k: v for k, v in art.weights.items() if "lora_" not in k}
        lora_weights = {k: v for k, v in art.weights.items() if "lora_" in k}
        loaded = load_model_weights(transformer, branch_weights)
        merged = merge_lora_state(transformer, lora_weights) if lora_weights else 0

        say(f"built from spec: {loaded} branch tensors, {merged} LoRA pairs merged",
            flush=True)
    else:
        say("DENSE BASE render: no checkpoint -- no transform, no branch", flush=True)

    if not cfg.render.offload_dit:
        transformer.to(device)
    transformer.eval().requires_grad_(False)

    external_records = []
    for xl in cfg.external_loras:
        state, scale, ranks = load_external_lora(xl.path, xl.alpha)
        pairs = merge_lora_state(transformer, state, scale)

        say(f"external LoRA {xl.path}: {pairs} pairs merged (ranks {ranks}, "
            f"scale {scale:g})", flush=True)
        external_records.append({"path": os.path.abspath(xl.path), "pairs": pairs,
                                 "ranks": ranks, "scale": scale})

    # ---- overlays, in order ----
    is_hybrid = next(iter_hybrids(transformer), None) is not None
    if cfg.kernels.npu_resident_blocks and not is_hybrid:
        raise ValueError("kernels.npu_resident_blocks requires a hybrid checkpoint")
    if getattr(cfg.kernels, "npu_pinned_offload", False) and not is_hybrid:
        raise ValueError("kernels.npu_pinned_offload requires a hybrid checkpoint")

    for attn in iter_hybrids(transformer):
        attn.teacher_mode = cfg.behavior.teacher_mode
        for prm in attn.parameters():
            if prm.dtype == torch.float32:
                prm.data = prm.data.to(torch.bfloat16)

    softmax_backend = None
    if is_hybrid:
        if cfg.kernels.inference_kernels and is_cuda(device):
            set_inference_mode(transformer, True)
        elif cfg.kernels.inference_kernels:
            # The installed inference bodies contain CUDA Triton/Inductor kernels and
            # preallocated in-place state writes.  Keep them available for CUDA, while
            # using the eager autograd-safe implementation on Ascend NPU.
            say(f"inference kernels: eager fallback on {device_of(device).type}", flush=True)

        softmax_backend = set_softmax_backend(
            transformer, cfg.kernels.softmax_backend,
            device=device if softmax_device is None else softmax_device,
        )
        say(f"window softmax: {softmax_backend}", flush=True)

    ablated = (apply_ablation(transformer, dict(cfg.ablation.overrides))
               if cfg.ablation.enabled and cfg.ablation.overrides else [])
    if ablated:
        say(f"ABLATION ACTIVE: {ablated} -- this output is a study, not a sample",
            flush=True)

    fp8_linears = 0
    if cfg.precision.fp8.enabled:
        fp8_linears = convert_linear_to_fp8(
            transformer, skip_end_blocks=cfg.precision.fp8.skip_end_blocks)
        say(f"fp8: {fp8_linears} Linears quantised", flush=True)
        # The bf16 weights the conversion freed, about 20 GiB, stay in PyTorch's cache,
        # where only PyTorch can reuse them. NCCL allocates its buffers outside that cache
        # at the first collective of each communicator, so on an 80 GB card the eight-GPU
        # render ran out of memory in its first all-to-all. Hand them back to the driver.
        empty_cache(device)

    resident = 0
    pinned_state = None
    if cfg.render.offload_dit and install_offload:
        resident, pinned_state = install_inference_offload(
            transformer, device, cfg, say=say)

    return InferenceModel(transformer=transformer, vae=vae, audio_vae=audio_vae,
                          artifact=art, is_hybrid=is_hybrid, merged_lora_pairs=merged,
                          external_loras=external_records, ablated=ablated,
                          fp8_linears=fp8_linears, softmax_backend=softmax_backend,
                          npu_resident_blocks=resident,
                          npu_pinned_offload=pinned_state,
                          total_blocks=len(transformer.transformer_blocks))


def render_record(cfg, model: InferenceModel) -> Dict[str, Any]:
    """The record next to the mp4: checkpoint identity, the overlay, the git commit and
    the ACTUAL kernel state. Entrypoints add their own sections (timings, parallel)."""
    art = model.artifact
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()

    return {
        "checkpoint": resolve_weights(cfg.checkpoint) if cfg.checkpoint else None,
        "checkpoint_head_sha256": (checkpoint_head_sha256(resolve_weights(cfg.checkpoint))
                                   if cfg.checkpoint else None),
        "model_spec_base_hash": art.model_spec["base"]["config_hash"] if art else None,
        "git_commit": commit,
        "overlay": resolved_dict(cfg),
        "ablation_active": model.ablated,
        "external_loras": model.external_loras,
        "fp8_linears": model.fp8_linears,
        "flex_backend": flex_latch_state(),
        "softmax_backend": {"resolved": model.softmax_backend},
        "merged_lora_pairs": model.merged_lora_pairs,
        "npu_residency": {
            "requested_blocks": cfg.kernels.npu_resident_blocks,
            "resident_blocks": model.npu_resident_blocks,
            "total_blocks": model.total_blocks,
            "released_before_decode": model.resident_blocks_released,
            "release_seconds": model.resident_release_seconds,
        },
        "npu_pinned_offload": {
            "requested": bool(getattr(cfg.kernels, "npu_pinned_offload", False)),
            "enabled": model.npu_pinned_offload is not None,
            "status": ((model.npu_pinned_offload or {}).get("status")
                       or "disabled"),
            "fallback": (model.npu_pinned_offload or {}).get("fallback", False),
            "error": (model.npu_pinned_offload or {}).get("error"),
            **(model.npu_pinned_offload or {}),
        },
    }


def latents_path(cfg) -> str:
    """Where `render.save_latents` puts the pre-VAE tensors: mirrors the mp4's path
    under results/ into `render.latents_root`."""
    out_abs = os.path.abspath(cfg.render.out)
    rel_out = os.path.relpath(out_abs, os.path.abspath("results"))

    if rel_out == os.pardir or rel_out.startswith(os.pardir + os.sep):
        rel_out = os.path.basename(out_abs)

    return os.path.join(cfg.render.latents_root, rel_out + ".latents.pt")


def write_json(record: Dict[str, Any], path: str) -> None:
    import json

    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
