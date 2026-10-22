"""Generate the merged, TP-sharded weight cache for `infer_tp.py`.

    python -m src.inference.prepare_tp_cache --out <cache_dir> \
        --config configs/inference/8nfe.yaml checkpoint=... base_source=... \
        vae_source=... external_loras=...

The normal CPU assembly path runs once per rank (0, then 1), including every LoRA
merge. Tensor parallelism is installed on CPU, then only that rank's final state dict
is saved. The runtime builds a meta skeleton, installs the same layout, loads one
shard, and never calls `merge_lora_state`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from omegaconf import OmegaConf
from safetensors import __version__ as safetensors_version
from safetensors.torch import save_file

from src.checkpoints import checkpoint_head_sha256
from src.config import ConfigError, load_config
from src.config.inference import (InferenceConfig, validate_ablation, validate_kernels,
                                  validate_parallel)
from src.inference.utils.assemble import build_inference_model
from src.inference.utils.tensor_parallel import TensorParallelRuntime, install_tensor_parallel
from src.inference.utils.tp_cache import CACHE_DTYPE, CACHE_KIND, CACHE_WORLD_SIZE, FORMAT_VERSION
from src.paths import H3_BASE, resolve_weights

CACHE_FILES = ("manifest.json", "rank0.safetensors", "rank1.safetensors")
GIB = 1024 ** 3


def _yaml_has_key(path: str, key: str) -> bool:
    try:
        node = OmegaConf.load(path)
    except Exception:
        return False
    return OmegaConf.select(node, key, default=None) is not None


def _arg_supplies(rest, key: str) -> bool:
    for index, token in enumerate(rest):
        if token == "--config" and index + 1 < len(rest):
            if _yaml_has_key(rest[index + 1], key):
                return True
            continue
        if token.startswith("--config="):
            if _yaml_has_key(token.split("=", 1)[1], key):
                return True
            continue
        if token.startswith("--"):
            continue
        name = token.split("=", 1)[0].lstrip("+~")
        if name == key:
            return True
    return False


def _load_generator_config(rest):
    rest = list(rest)
    if not _arg_supplies(rest, "render.prompt_file"):
        rest.append("render.prompt_file=__tp_cache_unused__.pt")
    if not _arg_supplies(rest, "render.out"):
        rest.append("render.out=__tp_cache_unused__.mp4")
    cfg = load_config(InferenceConfig, rest)
    if not cfg.checkpoint:
        raise ConfigError("prepare_tp_cache requires a hybrid checkpoint (checkpoint=...)")

    cfg.render.offload_dit = False
    cfg.kernels.npu_resident_blocks = 0
    cfg.kernels.npu_pinned_offload = False
    cfg.precision.fp8.enabled = False
    cfg.parallel.npu_tensor_parallel = True
    cfg.parallel.npu_head_parallel = False
    cfg.kernels.softmax_backend = "ref"
    cfg.parallel.npu_tp_parity = "off"
    cfg.parallel.merge_device = "cpu"
    # Only satisfies TP's NPU-only validation; both builds below run on CPU.
    cfg.render.device = "npu:0"
    try:
        validate_ablation(cfg)
        validate_kernels(cfg)
        validate_parallel(cfg)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return cfg


def _prepare_out(path: str, force: bool) -> None:
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ConfigError(f"--out {path} exists and is not a directory")
        if os.listdir(path) and not force:
            raise ConfigError(
                f"--out {path} is not empty; pass --force to overwrite only "
                f"{', '.join(CACHE_FILES)}"
            )
        return
    os.makedirs(path)


def _clone_shared_tensors(state):
    groups = {}
    for name, tensor in state.items():
        if tensor.device.type == "meta" or tensor.untyped_storage().data_ptr() == 0:
            continue
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        groups.setdefault(key, []).append(name)
    shared = {name for names in groups.values() if len(names) > 1 for name in names}
    if not shared:
        return state
    return {name: (tensor.detach().clone() if name in shared else tensor)
            for name, tensor in state.items()}


def _save_state(state, path: str) -> None:
    contiguous = {name: (tensor if tensor.is_contiguous() else tensor.contiguous())
                  for name, tensor in state.items()}
    try:
        save_file(contiguous, path)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "share memory" not in message and "shared memory" not in message:
            raise
        print("shared tensor storage detected; cloning only the shared groups", flush=True)
        save_file(_clone_shared_tensors(contiguous), path)


def _atomic_save_state(state, final_path: str) -> None:
    temp = final_path + f".tmp.{os.getpid()}"
    try:
        _save_state(state, temp)
        os.replace(temp, final_path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def _atomic_write_json(payload, path: str) -> None:
    temp = path + f".tmp.{os.getpid()}"
    try:
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


def _manifest(cfg, model):
    art = model.artifact
    spec = art.model_spec
    base_source = cfg.base_source or spec["base"]["source"]
    checkpoint = os.path.abspath(resolve_weights(cfg.checkpoint))
    return {
        "format_version": FORMAT_VERSION,
        "kind": CACHE_KIND,
        "world_size": CACHE_WORLD_SIZE,
        "dtype": CACHE_DTYPE,
        "checkpoint": checkpoint,
        "head_sha256": checkpoint_head_sha256(checkpoint),
        "model_spec": spec,
        "metadata": art.metadata,
        "base_source": {
            "source": os.path.abspath(resolve_weights(base_source)),
            "subfolder": spec["base"]["subfolder"],
        },
        "vae_source": os.path.abspath(resolve_weights(cfg.vae_source or H3_BASE)),
        "external_loras": model.external_loras,
        "merged_lora_pairs": model.merged_lora_pairs,
        "rank_files": {str(rank): f"rank{rank}.safetensors"
                       for rank in range(CACHE_WORLD_SIZE)},
        "rank_file_bytes": {},
        "shard_info": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors_version,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute the merged two-NPU TP rank shards")
    parser.add_argument("--out", required=True,
                        help="cache directory (manifest.json + rank0/1.safetensors)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the three known cache files in a non-empty --out")
    args, rest = parser.parse_known_args(argv)

    cfg = _load_generator_config(rest)
    out = os.path.abspath(args.out)
    _prepare_out(out, args.force)

    total_started = time.perf_counter()
    shard_info = {}
    rank_bytes = {}
    manifest = None
    for rank in range(CACHE_WORLD_SIZE):
        print(f"rank {rank}: building and merging on CPU", flush=True)
        build_started = time.perf_counter()
        model = build_inference_model(cfg, "cpu", load_decoders=False, log=True,
                                      softmax_device=torch.device("cpu"))
        build_seconds = time.perf_counter() - build_started
        if manifest is None:
            manifest = _manifest(cfg, model)

        model.artifact.weights.clear()
        runtime = TensorParallelRuntime(rank, CACHE_WORLD_SIZE, torch.device("cpu"),
                                        parity="off")
        shard_started = time.perf_counter()
        shard_info[rank] = install_tensor_parallel(model.transformer, runtime)
        shard_seconds = time.perf_counter() - shard_started

        state = model.transformer.state_dict()
        final_path = os.path.join(out, f"rank{rank}.safetensors")
        save_started = time.perf_counter()
        _atomic_save_state(state, final_path)
        save_seconds = time.perf_counter() - save_started
        rank_bytes[rank] = os.path.getsize(final_path)
        print(
            f"rank {rank}: build {build_seconds:.1f}s, shard {shard_seconds:.1f}s, "
            f"save {save_seconds:.1f}s, {rank_bytes[rank] / GIB:.2f} GiB -> {final_path}",
            flush=True,
        )
        del state
        del model
        gc.collect()

    manifest["shard_info"] = {str(rank): shard_info[rank]
                              for rank in range(CACHE_WORLD_SIZE)}
    manifest["rank_file_bytes"] = {str(rank): rank_bytes[rank]
                                   for rank in range(CACHE_WORLD_SIZE)}
    _atomic_write_json(manifest, os.path.join(out, "manifest.json"))
    print(
        f"total {time.perf_counter() - total_started:.1f}s; "
        f"{sum(rank_bytes.values()) / GIB:.2f} GiB in {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
