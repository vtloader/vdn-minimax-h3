"""Precomputed merged, TP-sharded weight cache for the two-NPU TP entrypoint.

The generator (``src/inference/prepare_tp_cache.py``) runs the normal CPU assembly
and LoRA merge once per checkpoint/base/external-LoRA combination, installs the
weight TP layout, and writes one shard per rank. At runtime each rank builds the
transformer skeleton on ``meta``, installs the same TP layout without touching base
weights, loads only its own shard, and either moves it to the NPU or keeps it on CPU
for block offload. ``merge_lora_state`` never runs on this path.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

import torch
from diffusers import MiniMaxH3Transformer3DModel
from safetensors.torch import load_file
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

from src.checkpoints import CheckpointArtifact, checkpoint_head_sha256
from src.inference.utils.assemble import InferenceModel, apply_ablation, load_decoders
from src.inference.utils.tensor_parallel import install_tensor_parallel
from src.models.hybrid_transform import (apply_hybrid_attention_transform,
                                         set_softmax_backend)
from src.paths import resolve_weights
from src.utils.device import synchronize

FORMAT_VERSION = 1
CACHE_KIND = "vdn_tp_shard"
CACHE_DTYPE = "bfloat16"
CACHE_WORLD_SIZE = 2


def load_manifest(cache_dir: str) -> Dict[str, Any]:
    cache_dir = os.path.abspath(cache_dir)
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"TP cache manifest not found: {path}; generate it with "
            f"`python -m src.inference.prepare_tp_cache --out {cache_dir} ...`"
        )
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"TP cache manifest {path} must contain a JSON object")
    # Internal, not serialized: lets validate_manifest check the rank files.
    manifest["_cache_dir"] = cache_dir
    return manifest


def _require_rank_file(manifest: Dict[str, Any], cache_dir: str, rank: int) -> str:
    files = manifest.get("rank_files")
    if not isinstance(files, dict):
        raise ValueError("TP cache manifest is missing rank_files")
    name = files.get(str(rank))
    if not isinstance(name, str) or not name or os.path.basename(name) != name:
        raise ValueError(f"TP cache manifest has an invalid rank {rank} file name {name!r}")
    path = os.path.join(cache_dir, name)
    if not os.path.isfile(path):
        raise ValueError(f"TP cache rank file missing: {path}")
    expected = (manifest.get("rank_file_bytes") or {}).get(str(rank))
    if expected is not None and os.path.getsize(path) != int(expected):
        raise ValueError(
            f"TP cache rank file {path} is {os.path.getsize(path)} bytes; "
            f"manifest says {expected}"
        )
    return path


def validate_manifest(manifest: Dict[str, Any], cfg, cache_dir: str = None) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("TP cache manifest must be a JSON object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"TP cache format_version {manifest.get('format_version')!r} != "
            f"{FORMAT_VERSION}"
        )
    if manifest.get("kind") != CACHE_KIND:
        raise ValueError(
            f"TP cache kind {manifest.get('kind')!r} != {CACHE_KIND!r}"
        )
    if manifest.get("world_size") != CACHE_WORLD_SIZE:
        raise ValueError(
            f"TP cache world_size {manifest.get('world_size')!r} != {CACHE_WORLD_SIZE}"
        )
    if manifest.get("dtype") != CACHE_DTYPE:
        raise ValueError(
            f"TP cache dtype {manifest.get('dtype')!r} != {CACHE_DTYPE!r}"
        )
    if cfg.precision.dtype != CACHE_DTYPE:
        raise ValueError(
            f"TP cache requires precision.dtype={CACHE_DTYPE}; config has "
            f"{cfg.precision.dtype!r}"
        )
    if not cfg.checkpoint:
        raise ValueError("TP cache requires a hybrid checkpoint")

    checkpoint = os.path.abspath(resolve_weights(cfg.checkpoint))
    try:
        head_sha256 = checkpoint_head_sha256(checkpoint)
    except OSError as exc:
        raise ValueError(f"TP cache checkpoint unavailable: {checkpoint} ({exc})") from exc
    if head_sha256 != manifest.get("head_sha256"):
        raise ValueError(
            f"TP cache was built for checkpoint {manifest.get('checkpoint')!r} "
            f"(head_sha256 {manifest.get('head_sha256')}); configured checkpoint "
            f"{checkpoint} has head_sha256 {head_sha256}"
        )

    base_source = manifest.get("base_source")
    if not isinstance(base_source, dict) or not base_source.get("source"):
        raise ValueError("TP cache manifest is missing base_source")
    if cfg.base_source:
        wanted = os.path.abspath(resolve_weights(cfg.base_source))
        if wanted != base_source["source"]:
            raise ValueError(
                f"TP cache base_source {base_source['source']!r} does not match "
                f"configured base_source {wanted!r}"
            )
    if cfg.vae_source:
        wanted = os.path.abspath(resolve_weights(cfg.vae_source))
        cached = manifest.get("vae_source")
        if cached and wanted != cached:
            raise ValueError(
                f"TP cache vae_source {cached!r} does not match configured "
                f"vae_source {wanted!r}"
            )

    spec = manifest.get("model_spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("base"), dict):
        raise ValueError("TP cache manifest is missing model_spec.base")
    if not spec.get("transforms"):
        raise ValueError("TP cache model_spec carries no transforms; expected hybrid")
    shard_info = manifest.get("shard_info")
    if not isinstance(shard_info, dict) or any(
            str(rank) not in shard_info for rank in range(CACHE_WORLD_SIZE)):
        raise ValueError("TP cache manifest is missing per-rank shard_info")

    records = manifest.get("external_loras")
    wanted = [os.path.abspath(x.path) for x in cfg.external_loras]
    got = [r.get("path") for r in (records or [])]
    if wanted != got:
        raise ValueError(
            f"TP cache external_loras {got} do not match configured external_loras "
            f"{wanted}; the cache covers one exact LoRA combination"
        )

    cache_dir = cache_dir or manifest.get("_cache_dir")
    if cache_dir:
        for rank in range(CACHE_WORLD_SIZE):
            _require_rank_file(manifest, cache_dir, rank)


def _set_submodule_attr(module, name: str, value) -> None:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _materialize_nonpersistent_buffers(transformer) -> None:
    """Recompute buffers that are absent from the cached state dict by design.

    ``load_state_dict(assign=True)`` replaces every persistent tensor, but the
    non-persistent RoPE ``inv_freq`` buffer stays on ``meta``. It is tiny and derived
    entirely from the transformer config, so rebuild it instead of caching it.
    """
    meta_buffers = [(name, buffer) for name, buffer in transformer.named_buffers()
                    if buffer.device.type == "meta"]
    for name, _ in meta_buffers:
        if name != "rope.inv_freq":
            raise RuntimeError(
                f"TP cache left non-persistent meta buffer {name!r} without a "
                "materialization rule"
            )
        rope = MiniMaxH3RotaryPosEmbed(
            rope_freq_dim=transformer.config.rope_freq_dim,
            rope_theta=transformer.config.rope_theta,
        )
        _set_submodule_attr(transformer, name, rope.inv_freq)

    meta_left = [name for name, tensor in list(transformer.named_parameters())
                 + list(transformer.named_buffers()) if tensor.device.type == "meta"]
    if meta_left:
        raise RuntimeError(f"TP cache left meta tensors behind: {meta_left[:8]}")


def _build_meta_transformer(base: Dict[str, Any]) -> torch.nn.Module:
    config = MiniMaxH3Transformer3DModel.load_config(
        resolve_weights(base["source"]),
        subfolder=base.get("subfolder") or "",
        revision=base.get("revision") or None,
    )
    dtype_orig = MiniMaxH3Transformer3DModel._set_default_torch_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            transformer = MiniMaxH3Transformer3DModel.from_config(config)
    finally:
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)
    return transformer


def build_cached_model(cfg, runtime, cache_dir: str) -> Tuple[InferenceModel, Dict[str, Any], float]:
    started = time.perf_counter()
    manifest = load_manifest(cache_dir)
    validate_manifest(manifest, cfg, cache_dir=cache_dir)

    spec = manifest["model_spec"]
    transformer = _build_meta_transformer(spec["base"])
    apply_hybrid_attention_transform(transformer, spec["transforms"][0]["config"])
    transformer.eval().requires_grad_(False)

    install_tensor_parallel(transformer, runtime, allow_meta=True)
    # The generation-time CPU layout is the authoritative record; the meta
    # installation cannot measure real storage bytes.
    shard_info = dict((manifest.get("shard_info") or {}).get(str(runtime.rank)) or {})

    rank_file = _require_rank_file(manifest, os.path.abspath(cache_dir), runtime.rank)
    state = load_file(rank_file, device="cpu")
    transformer.load_state_dict(state, strict=True, assign=True)
    del state
    transformer.requires_grad_(False)
    _materialize_nonpersistent_buffers(transformer)

    # row_gather prepares its full Row buffers with HCCL before hooks are installed.
    # Other offloaded modes can keep the cached shard on the CPU from the outset.
    if not cfg.render.offload_dit or runtime.parity == "row_gather":
        transformer.to(runtime.device)
        synchronize(runtime.device)

    vae = audio_vae = None
    if runtime.is_main:
        vae, audio_vae = load_decoders(cfg.vae_source or manifest.get("vae_source"))

    ablated = (apply_ablation(transformer, dict(cfg.ablation.overrides))
               if cfg.ablation.enabled and cfg.ablation.overrides else [])
    artifact = CheckpointArtifact(
        kind="weights",
        model_spec=spec,
        weights={},
        metadata=manifest.get("metadata") or {},
    )
    model = InferenceModel(
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        artifact=artifact,
        is_hybrid=True,
        merged_lora_pairs=int(manifest.get("merged_lora_pairs") or 0),
        external_loras=list(manifest.get("external_loras") or []),
        ablated=ablated,
        softmax_backend=set_softmax_backend(
            transformer, cfg.kernels.softmax_backend, device=runtime.device),
        total_blocks=len(transformer.transformer_blocks),
        npu_resident_blocks=0,
        npu_pinned_offload=None,
    )
    return model, shard_info, time.perf_counter() - started
