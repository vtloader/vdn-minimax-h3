"""Opt-in NPU pinned-host block offload.

The public Diffusers ``apply_group_offloading(use_stream=True)`` refuses NPU, while
TorchNPU supports pinned host memory, streams, and asynchronous H2D.  This module
installs the same block-level grouping through the private
``_apply_group_offloading`` entry point, after pinning every CPU parameter/buffer in
place, and keeps the top-level "unmatched" group synchronous for the first forward.

It is deliberately opt-in: pinning the 65.7 GiB transformer reserves about 97 GiB of
host memory (the allocator rounds each tensor up to a power of two).
"""
from __future__ import annotations

import hashlib
import importlib
import time
from dataclasses import fields
from typing import Any, Dict, Optional

import torch

from src.utils.device import synchronize

GIB = 1024 ** 3
PINNED_RESERVE_FACTOR = 1.5
HOST_HEADROOM_BYTES = 8 * GIB
PINNED_RESERVE_NOTE = "1.5x model bytes + 8 GiB headroom"
_STATE_ATTR = "_npu_pinned_offload_state"


def _npu_backend():
    backend = getattr(torch, "npu", None)
    if backend is None:
        try:
            importlib.import_module("torch_npu")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "kernels.npu_pinned_offload requires torch-npu; install a torch-npu "
                "build matching this PyTorch and CANN version"
            ) from exc
        backend = getattr(torch, "npu", None)
    if backend is None:
        raise RuntimeError(
            "kernels.npu_pinned_offload requires torch.npu; torch-npu did not register it"
        )
    if not hasattr(backend, "Stream"):
        raise RuntimeError(
            "kernels.npu_pinned_offload requires torch.npu.Stream (TorchNPU >= 2.7)"
        )
    return backend


def _group_offloading_api():
    module = importlib.import_module("diffusers.hooks.group_offloading")
    required = (
        "GroupOffloadingConfig",
        "GroupOffloadingType",
        "ModuleGroup",
        "_apply_group_offloading",
        "_GROUP_OFFLOADING",
        "_LAZY_PREFETCH_GROUP_OFFLOADING",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "kernels.npu_pinned_offload requires the vendored Diffusers block-level "
            f"offload API; missing {missing}. Refusing to fall back to pageable offload."
        )
    config_fields = {entry.name for entry in fields(module.GroupOffloadingConfig)}
    required_fields = {"onload_device", "offload_device", "offload_type", "non_blocking",
                       "record_stream", "low_cpu_mem_usage", "num_blocks_per_group",
                       "offload_to_disk_path", "stream", "block_modules", "exclude_kwargs",
                       "module_prefix"}
    missing_fields = sorted(required_fields - config_fields)
    if missing_fields:
        raise RuntimeError(
            "kernels.npu_pinned_offload requires GroupOffloadingConfig fields "
            f"{missing_fields}; the installed Diffusers version is incompatible"
        )
    if not hasattr(module.ModuleGroup, "_init_cpu_param_dict"):
        raise RuntimeError(
            "kernels.npu_pinned_offload requires ModuleGroup._init_cpu_param_dict; "
            "the installed Diffusers version is incompatible"
        )
    return module


def read_mem_available_bytes() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _host_memory_snapshot(backend) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"available_bytes": read_mem_available_bytes()}
    try:
        stats = backend.memory.host_memory_stats()
        keys = ("allocation.current", "allocation.peak", "reserved_bytes.current",
                "reserved_bytes.peak", "allocated_bytes.current", "allocated_bytes.peak",
                "num_host_alloc", "num_host_free", "host_alloc_time.total",
                "host_alloc_time.count")
        snapshot["pinned"] = {key: stats[key] for key in keys if key in stats}
    except Exception as exc:
        snapshot["pinned"] = {"error": f"{type(exc).__name__}: {exc}"}
    return snapshot


def _cpu_tensor_bytes(transformer: torch.nn.Module) -> int:
    total = 0
    for tensor in list(transformer.parameters()) + list(transformer.buffers()):
        if tensor.device.type == "cpu":
            total += tensor.numel() * tensor.element_size()
    return total


def _unpinned_cpu_tensor_bytes(transformer: torch.nn.Module) -> int:
    total = 0
    for tensor in list(transformer.parameters()) + list(transformer.buffers()):
        if tensor.device.type == "cpu" and not tensor.data.is_pinned():
            total += tensor.numel() * tensor.element_size()
    return total


def _weight_fingerprint(transformer: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in list(transformer.named_parameters()) + list(transformer.named_buffers()):
        data = tensor.data
        digest.update(name.encode("utf-8"))
        digest.update(f":{id(tensor)}:{data.data_ptr()}:{tensor._version}:"
                      f"{tuple(data.shape)}:{data.dtype}".encode("utf-8"))
    return digest.hexdigest()


def pin_model_in_place(transformer: torch.nn.Module) -> Dict[str, Any]:
    """Pin every CPU parameter/buffer in place; ``pin_memory`` is idempotent."""
    started = time.perf_counter()
    tensors = 0
    pinned_bytes = 0
    for tensor in list(transformer.parameters()) + list(transformer.buffers()):
        if tensor.device.type != "cpu" or tensor.data.is_pinned():
            continue
        nbytes = tensor.numel() * tensor.element_size()
        tensor.data = tensor.data.pin_memory()
        tensors += 1
        pinned_bytes += nbytes
    return {
        "tensors_pinned": tensors,
        "pinned_bytes": pinned_bytes,
        "pin_seconds": time.perf_counter() - started,
    }


def _top_level_group(transformer: torch.nn.Module, module):
    registry = getattr(transformer, "_diffusers_hook", None)
    hook = registry.get_hook(module._GROUP_OFFLOADING) if registry is not None else None
    if hook is None:
        raise RuntimeError(
            "kernels.npu_pinned_offload: Diffusers did not attach a top-level "
            "group-offloading hook; refusing to continue with possibly pageable weights"
        )
    return hook.group


def _validate_install(transformer: torch.nn.Module, device: str,
                      state: Dict[str, Any], module) -> None:
    if state.get("device") != str(device):
        raise RuntimeError(
            f"kernels.npu_pinned_offload is already installed on {state.get('device')}, "
            f"not {device}"
        )
    unpinned = _unpinned_cpu_tensor_bytes(transformer)
    if unpinned:
        raise RuntimeError(
            "kernels.npu_pinned_offload: transformer weights changed after installation "
            f"({unpinned / GIB:.2f} GiB unpinned); uninstall and reinstall the offload"
        )
    if _weight_fingerprint(transformer) != state["weight_fingerprint"]:
        raise RuntimeError(
            "kernels.npu_pinned_offload: transformer weights changed after installation; "
            "uninstall and reinstall the offload"
        )
    _top_level_group(transformer, module)


def apply_npu_pinned_offload(transformer: torch.nn.Module, device: str, *,
                             reserve_factor: float = PINNED_RESERVE_FACTOR,
                             headroom_bytes: int = HOST_HEADROOM_BYTES) -> Dict[str, Any]:
    """Install pinned-host async block offload; idempotent for an unchanged model."""
    resolved = torch.device(device)
    if resolved.type != "npu":
        raise ValueError(
            f"kernels.npu_pinned_offload is NPU-only; got device={device!r}"
        )
    backend = _npu_backend()
    module = _group_offloading_api()

    existing = getattr(transformer, _STATE_ATTR, None)
    if existing is not None:
        _validate_install(transformer, device, existing, module)
        return existing

    model_bytes = _cpu_tensor_bytes(transformer)
    if model_bytes == 0:
        raise RuntimeError(
            "kernels.npu_pinned_offload expects CPU-resident transformer weights; "
            "load with render.offload_dit=true"
        )

    before = _host_memory_snapshot(backend)
    available = before.get("available_bytes")
    required = int(reserve_factor * model_bytes) + int(headroom_bytes)
    if available is None:
        raise RuntimeError(
            "kernels.npu_pinned_offload cannot read MemAvailable from /proc/meminfo; "
            "refusing to pin host weights without a host-memory guard"
        )
    if available < required:
        raise RuntimeError(
            "kernels.npu_pinned_offload needs "
            f"{required / GIB:.1f} GiB host MemAvailable ({PINNED_RESERVE_NOTE} for "
            f"{model_bytes / GIB:.1f} GiB of weights) but only "
            f"{available / GIB:.1f} GiB is available"
        )

    pinned = pin_model_in_place(transformer)
    after_pin = _host_memory_snapshot(backend)

    stream = backend.Stream()
    config = module.GroupOffloadingConfig(
        onload_device=resolved,
        offload_device=torch.device("cpu"),
        offload_type=module.GroupOffloadingType.BLOCK_LEVEL,
        non_blocking=True,
        record_stream=False,
        low_cpu_mem_usage=False,
        num_blocks_per_group=1,
        offload_to_disk_path=None,
        stream=stream,
        block_modules=getattr(transformer, "_group_offload_block_modules", None),
        exclude_kwargs=getattr(transformer, "_skip_keys", None),
        module_prefix="",
    )

    started = time.perf_counter()
    module._apply_group_offloading(transformer, config)
    apply_seconds = time.perf_counter() - started

    top_group = _top_level_group(transformer, module)
    top_group.stream = stream
    top_group.non_blocking = False
    top_group.record_stream = False
    top_group.cpu_param_dict = top_group._init_cpu_param_dict()

    synchronize(device)
    after_install = _host_memory_snapshot(backend)

    def pinned_bytes(snapshot: Dict[str, Any], key: str):
        return snapshot.get("pinned", {}).get(key)

    cache_before = pinned_bytes(before, "reserved_bytes.current") or 0
    cache_after = pinned_bytes(after_install, "reserved_bytes.current") or 0
    after_available = after_install.get("available_bytes")

    state = {
        "status": "installed",
        "fallback": False,
        "device": str(device),
        "diffusers_version": getattr(importlib.import_module("diffusers"), "__version__", None),
        "private_api": "diffusers.hooks.group_offloading._apply_group_offloading",
        "model_bytes": model_bytes,
        "model_gib": round(model_bytes / GIB, 4),
        "tensors_pinned": pinned["tensors_pinned"],
        "pinned_bytes": pinned["pinned_bytes"],
        "pin_seconds": round(pinned["pin_seconds"], 4),
        "apply_seconds": round(apply_seconds, 4),
        "host_required_bytes": required,
        "host_required_gib": round(required / GIB, 2),
        "host_available_before_bytes": available,
        "host_available_after_pin_bytes": after_pin.get("available_bytes"),
        "host_available_after_install_bytes": after_available,
        "host_pin_delta_bytes": (available - after_pin["available_bytes"]
                                 if after_pin.get("available_bytes") is not None else None),
        "host_pinned_delta_bytes": cache_after - cache_before,
        "pinned_cache_bytes": cache_after,
        "host_memory_after_install": after_install,
        "stream": True,
        "non_blocking": True,
        "record_stream": False,
        "num_blocks_per_group": 1,
        "top_level_non_blocking": False,
        "first_forward_synchronous": True,
        "error": None,
        "weight_fingerprint": _weight_fingerprint(transformer),
    }
    setattr(transformer, _STATE_ATTR, state)
    return state


def uninstall_npu_pinned_offload(transformer: torch.nn.Module, device: str) -> bool:
    """Remove the offload hooks.  Pinned weights stay pinned until the process exits."""
    state = getattr(transformer, _STATE_ATTR, None)
    if state is None:
        return False
    module = _group_offloading_api()
    synchronize(device)
    groups = {}
    for submodule in transformer.modules():
        registry = getattr(submodule, "_diffusers_hook", None)
        if registry is None:
            continue
        hook = registry.get_hook(module._GROUP_OFFLOADING)
        if hook is not None:
            groups[id(hook.group)] = hook.group
    for group in groups.values():
        group.offload_()
    synchronize(device)
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None:
        registry.remove_hook(module._LAZY_PREFETCH_GROUP_OFFLOADING, recurse=True)
        registry.remove_hook(module._GROUP_OFFLOADING, recurse=True)
    delattr(transformer, _STATE_ATTR)
    return True
