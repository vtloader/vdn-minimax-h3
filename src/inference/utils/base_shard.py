"""Load one rank's final TP shard directly from the base shards and branch tensor.

``parallel.merge_device="npu"`` uses this instead of assembling a complete merged
transformer on CPU.  The skeleton is built on ``meta``, tensor parallelism is
installed exactly as on the CPU/cache paths, and every final parameter is read
only in the shape that rank keeps:

* base tensors come from the 14 safetensors files named by the base index;
* checkpoint branch tensors come from the exploded artifact's
  ``linear_branch/model.safetensors`` (already loaded by ``load_checkpoint``);
* ``ColumnLinear`` reads output rows, ``RowLinear`` reads input columns;
* directly sharded branch tensors (``alpha.dt_bias``, ``alpha.A_log`` and the
  short-conv weights) read one contiguous first-dimension interval;
* replicated parameters read the complete tensor.

No merged CPU model and no ``merge_lora_state`` call exist on this path.  The
checkpoint's own LoRA pairs and every external adapter are folded into the
NPU-resident shards with ``merge_lora_state_npu``.
"""
from __future__ import annotations

import json
import math
import mmap
import os
import struct

import numpy as np
import time
from typing import Any, Dict, Iterable, Tuple

import torch

from src.checkpoints import CheckpointArtifact, load_checkpoint
from src.inference.utils.assemble import (InferenceModel, apply_ablation,
                                          load_decoders)
from src.inference.utils.lora import load_external_lora, merge_lora_state_npu
from src.inference.utils.tensor_parallel import (ColumnLinear, RowLinear,
                                                 install_tensor_parallel)
from src.inference.utils.tp_cache import (_build_meta_transformer,
                                          _materialize_nonpersistent_buffers)
from src.models.hybrid_transform import (apply_hybrid_attention_transform,
                                         set_softmax_backend)
from src.paths import resolve_weights
from src.utils.device import synchronize

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_NP_DTYPES = {
    "BF16": (np.uint16, torch.bfloat16),
    "F16": (np.float16, torch.float16),
    "F32": (np.float32, torch.float32),
    "F64": (np.float64, torch.float64),
    "I64": (np.int64, torch.int64),
    "I32": (np.int32, torch.int32),
    "I16": (np.int16, torch.int16),
    "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8),
    "BOOL": (np.bool_, torch.bool),
}
_INDEX_NAMES = ("diffusion_pytorch_model.safetensors.index.json",
                "model.safetensors.index.json")


def _checked_slice(rows: slice, size: int, where: str) -> Tuple[int, int]:
    start = 0 if rows.start is None else rows.start
    stop = size if rows.stop is None else rows.stop
    step = 1 if rows.step is None else rows.step
    if step != 1:
        raise ValueError(f"{where}: only step=1 slices are supported, got {rows!r}")
    if start < 0 or stop < 0 or start > stop or stop > size:
        raise ValueError(f"{where}: slice {rows!r} outside 0..{size}")
    return start, stop


def _contiguous_runs(indices: Iterable[int], size: int, where: str):
    values = [int(index) for index in indices]
    if not values:
        return []
    for index in values:
        if index < 0 or index >= size:
            raise ValueError(f"{where}: row index {index} outside 0..{size}")
    runs = []
    start = previous = values[0]
    for index in values[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous + 1))
        start = previous = index
    runs.append((start, previous + 1))
    return runs


class SafeTensorReader:
    """Read selected tensor data from one safetensors file through ``mmap``.

    Every returned tensor is a contiguous CPU copy with its own storage, so the
    reader (and its file) may be closed as soon as the call returns.  Column and
    row selections copy only the requested bytes; a tensor-index row selection
    is split into contiguous runs (SwiGLU's two FFN halves are two runs).
    """

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._file = open(self.path, "rb")
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise
        try:
            if len(self._mmap) < 8:
                raise ValueError(f"{self.path}: safetensors file is shorter than its header")
            header_len = struct.unpack("<Q", self._mmap[:8])[0]
            data_start = 8 + header_len
            if data_start > len(self._mmap):
                raise ValueError(f"{self.path}: safetensors header length {header_len} "
                                 "exceeds the file")
            header = json.loads(self._mmap[8:data_start].decode("utf-8"))
            if not isinstance(header, dict):
                raise ValueError(f"{self.path}: safetensors header is not an object")
            self._data_start = data_start
            self._infos = {}
            for name, raw in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(raw, dict):
                    raise ValueError(f"{self.path}: tensor {name!r} has no header object")
                self._infos[name] = self._validate_info(name, raw)
        except Exception:
            self._mmap.close()
            self._file.close()
            raise

    def _validate_info(self, name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dtype_name = raw["dtype"]
            shape = tuple(int(dim) for dim in raw["shape"])
            start, stop = (int(x) for x in raw["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{self.path}: tensor {name!r} has an invalid header") from exc
        dtype = _DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(f"{self.path}: tensor {name!r} has unsupported dtype "
                             f"{dtype_name!r}")
        if any(dim < 0 for dim in shape):
            raise ValueError(f"{self.path}: tensor {name!r} has a negative dimension "
                             f"{shape}")
        if start < 0 or stop < start or self._data_start + stop > len(self._mmap):
            raise ValueError(f"{self.path}: tensor {name!r} data_offsets {start, stop} "
                             "outside the file")
        expected = math.prod(shape) * dtype.itemsize
        if stop - start != expected:
            raise ValueError(
                f"{self.path}: tensor {name!r} shape {shape} with {dtype_name} needs "
                f"{expected} bytes, data_offsets span {stop - start}")
        return {"dtype": dtype, "dtype_name": dtype_name, "shape": shape,
                "data_offsets": (start, stop)}

    def keys(self):
        return tuple(self._infos)

    def __contains__(self, name):
        return name in self._infos

    def tensor_info(self, name: str) -> Dict[str, Any]:
        if name not in self._infos:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        info = self._infos[name]
        return {"dtype": info["dtype"], "dtype_name": info["dtype_name"],
                "shape": info["shape"], "data_offsets": info["data_offsets"]}

    def _buffer(self, info: Dict[str, Any], byte_offset: int, count: int):
        dtype = info["dtype"]
        start, stop = info["data_offsets"]
        nbytes = count * dtype.itemsize
        if byte_offset < 0 or byte_offset + nbytes > stop - start:
            raise ValueError(f"{self.path}: read [{byte_offset}, {byte_offset + nbytes}) "
                             f"outside tensor data span {stop - start}")
        if nbytes == 0:
            return torch.empty(0, dtype=dtype)
        return torch.frombuffer(self._mmap, dtype=torch.uint8, count=nbytes,
                                offset=self._data_start + start + byte_offset).view(dtype)

    def read_full(self, name: str) -> torch.Tensor:
        info = self._infos.get(name)
        if info is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        shape = info["shape"]
        out = torch.empty(shape, dtype=info["dtype"])
        if out.numel():
            out.reshape(-1).copy_(self._buffer(info, 0, out.numel()))
        return out

    def read(self, name: str, *, rows=None, columns=None) -> torch.Tensor:
        """Read ``rows`` (first-dimension selection) or ``columns`` (2-D only)."""
        if rows is not None and columns is not None:
            raise ValueError("rows and columns cannot be selected together")
        if columns is not None:
            return self.read_columns(name, columns)
        if rows is not None:
            return self.read_rows(name, rows)
        return self.read_full(name)

    def read_first_dim(self, name: str, rows) -> torch.Tensor:
        return self.read_rows(name, rows)

    def read_rows(self, name: str, rows) -> torch.Tensor:
        info = self._infos.get(name)
        if info is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        shape = info["shape"]
        if not shape:
            raise ValueError(f"{self.path}: tensor {name!r} is scalar; no rows to select")
        dtype, dim0 = info["dtype"], shape[0]
        if isinstance(rows, slice):
            start, stop = _checked_slice(rows, dim0, f"{self.path}: {name}")
            runs = [(start, stop)]
            count = stop - start
        else:
            runs = _contiguous_runs(rows, dim0, f"{self.path}: {name}")
            count = sum(stop - start for start, stop in runs)
        out = torch.empty((count,) + shape[1:], dtype=dtype)
        row_elems = math.prod(shape[1:])
        row_bytes = row_elems * dtype.itemsize
        offset = 0
        for start, stop in runs:
            length = stop - start
            if not length:
                continue
            raw = self._buffer(info, start * row_bytes, length * row_elems)
            out[offset:offset + length].copy_(
                raw.reshape((length,) + shape[1:]))
            offset += length
        return out

    def read_columns(self, name: str, columns) -> torch.Tensor:
        info = self._infos.get(name)
        if info is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        shape = info["shape"]
        if len(shape) != 2:
            raise ValueError(f"{self.path}: tensor {name!r} is {len(shape)}-D; "
                             "column selection requires 2-D")
        if not isinstance(columns, slice):
            raise ValueError(f"{self.path}: columns must be a slice, got {columns!r}")
        start, stop = _checked_slice(columns, shape[1], f"{self.path}: {name}")
        np_dtype, torch_dtype = _NP_DTYPES[info["dtype_name"]]
        width = stop - start
        # NumPy's strided source view touches only the selected column pages;
        # ascontiguousarray performs the copy in one vectorised loop.
        full = np.frombuffer(
            self._mmap, dtype=np_dtype, count=math.prod(shape),
            offset=self._data_start + info["data_offsets"][0],
        ).reshape(shape)
        sub = np.array(full[:, start:stop], copy=True, order="C")
        out = torch.from_numpy(sub)
        return out.view(torch_dtype) if torch_dtype != out.dtype else out

    def close(self):
        mmap_obj = getattr(self, "_mmap", None)
        if mmap_obj is not None:
            self._mmap = None
            mmap_obj.close()
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            self._file = None
            file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _base_weight_map(base: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    root = resolve_weights(base["source"])
    folder = os.path.join(root, base.get("subfolder") or "")
    for name in _INDEX_NAMES:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                weight_map = (json.load(f) or {}).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"{path}: missing a non-empty weight_map")
            bad = [key for key, shard in weight_map.items()
                   if not isinstance(key, str) or not isinstance(shard, str)
                   or os.path.basename(shard) != shard]
            if bad:
                raise ValueError(f"{path}: invalid weight_map entries {bad[:4]}")
            return folder, weight_map

    single = os.path.join(folder, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(single):
        with SafeTensorReader(single) as reader:
            return folder, {name: os.path.basename(single) for name in reader.keys()}
    raise FileNotFoundError(
        f"{folder}: no diffusion_pytorch_model.safetensors.index.json, "
        "model.safetensors.index.json or single diffusion_pytorch_model.safetensors")


def _branch_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(tensor.shape)


def _select_branch(source: torch.Tensor, module, name: str) -> torch.Tensor:
    if isinstance(module, ColumnLinear) and name.endswith((".weight", ".bias")):
        rows = module.rows
        if isinstance(rows, slice):
            return source[rows].clone().contiguous()
        return source[rows].contiguous()
    if isinstance(module, RowLinear) and name.endswith(".weight"):
        return source[:, module.columns].clone().contiguous()
    return source


def _first_dim_shard(shape, target_shape, rank: int, world_size: int):
    if (len(shape) == len(target_shape) and shape[1:] == target_shape[1:]
            and shape[0] == target_shape[0] * world_size):
        step = shape[0] // world_size
        return slice(rank * step, (rank + 1) * step)
    return None


def build_base_shard_model(cfg, runtime, *, log: bool = True):
    """Build one rank's final TP model and return ``(model, info, seconds)``.

    The base model is never materialised: the hybrid skeleton lives on ``meta``
    until this rank's exact shard is read from disk.  The checkpoint's branch
    tensors and LoRA pairs are the only full tensors loaded on the host (the
    exploded artifact's 5 GiB branch/adapter files).
    """
    say = print if log else (lambda *a, **k: None)
    started = time.perf_counter()
    if not cfg.checkpoint:
        raise ValueError("parallel.merge_device='npu' requires a hybrid checkpoint")
    if getattr(cfg.parallel, "merge_device", "cpu") != "npu":
        raise ValueError("build_base_shard_model is only for parallel.merge_device='npu'")

    checkpoint_started = time.perf_counter()
    art = load_checkpoint(resolve_weights(cfg.checkpoint))
    if art.metadata.get("truncated_blocks"):
        raise RuntimeError(f"{cfg.checkpoint} is a truncated smoke-test artifact")
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    spec = art.model_spec
    base = dict(spec["base"])
    if cfg.base_source:
        base["source"] = cfg.base_source
    transforms = spec.get("transforms") or []
    if len(transforms) != 1 or transforms[0].get("type") != "hybrid_attention":
        raise ValueError("parallel.merge_device='npu' requires one hybrid_attention transform")

    transformer = _build_meta_transformer(base)
    apply_hybrid_attention_transform(transformer, transforms[0]["config"])
    transformer.eval().requires_grad_(False)
    install_tensor_parallel(transformer, runtime, allow_meta=True)

    folder, weight_map = _base_weight_map(base)
    branch_keys = {name for name in art.weights if "lora_" not in name}
    modules = dict(transformer.named_modules())
    expected = transformer.state_dict()

    readers: Dict[str, SafeTensorReader] = {}

    def reader_for(filename: str) -> SafeTensorReader:
        reader = readers.get(filename)
        if reader is None:
            reader = SafeTensorReader(os.path.join(folder, filename))
            readers[filename] = reader
        return reader

    state = {}
    used_branch = set()
    missing = []
    base_read_seconds = 0.0
    branch_read_seconds = 0.0
    try:
        for name, target in expected.items():
            target_shape = tuple(target.shape)
            module_name = (name.rsplit(".", 1)[0]
                           if name.endswith((".weight", ".bias")) else None)
            module = modules.get(module_name) if module_name else None

            if name in branch_keys:
                source = art.weights.pop(name)
                used_branch.add(name)
                branch_started = time.perf_counter()
                source_shape = _branch_shape(source)
                if isinstance(module, (ColumnLinear, RowLinear)):
                    value = _select_branch(source, module, name)
                elif source_shape == target_shape:
                    value = source
                else:
                    interval = _first_dim_shard(source_shape, target_shape,
                                                runtime.rank, runtime.world_size)
                    if interval is None:
                        raise ValueError(
                            f"{name}: branch shape {source_shape} does not match final "
                            f"{target_shape} and is not a first-dimension halving")
                    value = source[interval].clone().contiguous()
                del source
                branch_read_seconds += time.perf_counter() - branch_started
            else:
                base_key = name.replace(".attn.orig.", ".attn.", 1)
                filename = weight_map.get(base_key)
                if filename is None:
                    missing.append((name, base_key))
                    continue
                reader = reader_for(filename)
                source_shape = reader.tensor_info(base_key)["shape"]
                read_started = time.perf_counter()
                if isinstance(module, ColumnLinear) and name.endswith(".weight"):
                    value = reader.read_rows(base_key, module.rows)
                elif isinstance(module, ColumnLinear) and name.endswith(".bias"):
                    value = reader.read_rows(base_key, module.rows)
                elif isinstance(module, RowLinear) and name.endswith(".weight"):
                    value = reader.read_columns(base_key, module.columns)
                elif isinstance(module, RowLinear) and name.endswith(".bias"):
                    value = reader.read_full(base_key)
                elif source_shape == target_shape:
                    value = reader.read_full(base_key)
                else:
                    interval = _first_dim_shard(source_shape, target_shape,
                                                runtime.rank, runtime.world_size)
                    if interval is None:
                        raise ValueError(
                            f"{name}: base tensor {base_key} shape {source_shape} does not "
                            f"match final {target_shape} and is not a first-dimension halving")
                    value = reader.read_first_dim(base_key, interval)
                base_read_seconds += time.perf_counter() - read_started

            if tuple(value.shape) != target_shape:
                raise ValueError(
                    f"{name}: loaded shape {tuple(value.shape)} != final {target_shape} "
                    f"(source {source_shape})")
            state[name] = value
    finally:
        for reader in readers.values():
            reader.close()

    if missing:
        raise RuntimeError(f"{len(missing)} final parameters have no base tensor, e.g. "
                           f"{missing[:8]}")
    unused = sorted(branch_keys - used_branch)
    if unused:
        raise RuntimeError(f"{len(unused)} branch tensors did not reach the final model, "
                           f"e.g. {unused[:8]}")
    expected_keys = set(expected)
    missing = [name for name in expected_keys if name not in state]
    extra = [name for name in state if name not in expected_keys]
    if missing or extra:
        raise RuntimeError(f"final parameter set mismatch: missing {missing[:8]}, "
                           f"extra {extra[:8]}")
    del expected

    assign_started = time.perf_counter()
    with torch.no_grad():
        transformer.load_state_dict(state, strict=True, assign=True)
        transformer.requires_grad_(False)
        _materialize_nonpersistent_buffers(transformer)
    assign_seconds = time.perf_counter() - assign_started
    state_bytes = sum(tensor.numel() * tensor.element_size()
                      for tensor in state.values())
    del state

    onload_started = time.perf_counter()
    transformer.to(runtime.device)
    synchronize(runtime.device)
    onload_seconds = time.perf_counter() - onload_started

    merge_started = time.perf_counter()
    lora_state = {name: tensor for name, tensor in art.weights.items()
                  if "lora_" in name}
    with torch.no_grad():
        merged = (merge_lora_state_npu(
            transformer, lora_state,
            delta_device=getattr(cfg.parallel, "merge_delta_device", "npu"))
                  if lora_state else 0)
    del lora_state
    external_records = []
    for xl in cfg.external_loras:
        external_state, scale, ranks = load_external_lora(xl.path, xl.alpha)
        pairs = merge_lora_state_npu(
            transformer, external_state, scale,
            delta_device=getattr(cfg.parallel, "merge_delta_device", "npu"))
        say(f"external LoRA {xl.path}: {pairs} pairs merged (ranks {ranks}, "
            f"scale {scale:g})", flush=True)
        external_records.append({"path": os.path.abspath(xl.path), "pairs": pairs,
                                 "ranks": ranks, "scale": scale})
        del external_state
    npu_merge_seconds = time.perf_counter() - merge_started
    art.weights.clear()

    decoder_started = time.perf_counter()
    vae = audio_vae = None
    if runtime.is_main:
        vae, audio_vae = load_decoders(cfg.vae_source)
    decoder_seconds = time.perf_counter() - decoder_started

    ablated = (apply_ablation(transformer, dict(cfg.ablation.overrides))
               if cfg.ablation.enabled and cfg.ablation.overrides else [])
    artifact = CheckpointArtifact(kind=art.kind, model_spec=art.model_spec,
                                  weights={}, metadata=art.metadata)
    model = InferenceModel(
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        artifact=artifact,
        is_hybrid=True,
        merged_lora_pairs=merged,
        external_loras=external_records,
        ablated=ablated,
        softmax_backend=set_softmax_backend(
            transformer, cfg.kernels.softmax_backend, device=runtime.device),
        total_blocks=len(transformer.transformer_blocks),
        npu_resident_blocks=0,
        npu_pinned_offload=None,
    )
    seconds = time.perf_counter() - started
    info = {
        "checkpoint": os.path.abspath(resolve_weights(cfg.checkpoint)),
        "base_source": os.path.abspath(resolve_weights(base["source"])),
        "checkpoint_seconds": checkpoint_seconds,
        "base_read_seconds": base_read_seconds,
        "branch_read_seconds": branch_read_seconds,
        "assign_seconds": assign_seconds,
        "onload_seconds": onload_seconds,
        "npu_merge_seconds": npu_merge_seconds,
        "merge_delta_device": getattr(cfg.parallel, "merge_delta_device", "npu"),
        "decoder_seconds": decoder_seconds,
        "base_shard_seconds": seconds,
        "state_bytes": state_bytes,
        "state_parameters": len(expected_keys),
        "merged_lora_pairs": merged,
        "external_loras": external_records,
    }
    say(f"base shard: {state_bytes / (1024 ** 3):.2f} GiB, {len(expected_keys)} parameters, "
        f"read {base_read_seconds + branch_read_seconds:.1f}s, assign {assign_seconds:.1f}s, "
        f"onload {onload_seconds:.1f}s, NPU merge {npu_merge_seconds:.1f}s "
        f"({merged} checkpoint pairs)", flush=True)
    return model, info, seconds
