"""Device dispatch for CUDA, Ascend NPU, and generic CPU helpers.

Generic helpers treat ``None`` as CPU. Distributed entry points use ``local_device``
to select CUDA first, then Ascend NPU, and require an available accelerator.
"""
from __future__ import annotations

import contextlib
import importlib
import os
from typing import Iterator

import torch


def _ensure_npu():
    """Return the NPU backend, importing its extension only when needed."""
    backend = getattr(torch, "npu", None)
    if backend is None:
        try:
            importlib.import_module("torch_npu")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Ascend NPU support requires torch-npu. Install a torch-npu build "
                "matching your PyTorch and CANN versions."
            ) from exc
        backend = getattr(torch, "npu", None)
    if backend is None:
        raise RuntimeError(
            "torch-npu did not register torch.npu. Install a torch-npu build "
            "matching your PyTorch and CANN versions."
        )
    return backend


def device_of(device=None) -> torch.device:
    """Normalize a CPU/CUDA/NPU string, device, or tensor; None means CPU."""
    if isinstance(device, torch.Tensor):
        device = device.device
    if device is not None and str(device).split(":", 1)[0] == "npu":
        _ensure_npu()
    try:
        resolved = torch.device(device) if device is not None else torch.device("cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Invalid device {device!r}; expected cpu, cuda[:index], or npu[:index]."
        ) from exc
    if resolved.type not in {"cpu", "cuda", "npu"}:
        raise ValueError(f"Unsupported device type {resolved.type!r}; expected cpu, cuda, or npu.")
    return resolved


def is_cuda(device=None) -> bool:
    return device_of(device).type == "cuda"


def is_npu(device=None) -> bool:
    return device_of(device).type == "npu"


def distributed_backend(device=None) -> str:
    """Return NCCL for CUDA or HCCL for NPU; CPU training is unsupported."""
    resolved = local_device() if device is None else device_of(device)
    if resolved.type == "cuda":
        return "nccl"
    if resolved.type == "npu":
        return "hccl"
    raise ValueError(f"Distributed training requires a CUDA or Ascend NPU device; got {resolved}.")


def local_device() -> torch.device:
    """Resolve torchrun's LOCAL_RANK, preferring available CUDA over NPU."""
    value = os.environ.get("LOCAL_RANK", "0")
    try:
        rank = int(value)
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be a nonnegative integer; got {value!r}.") from exc
    if rank < 0:
        raise ValueError(f"LOCAL_RANK must be a nonnegative integer; got {value!r}.")
    if torch.cuda.is_available():
        return torch.device("cuda", rank)
    try:
        backend = _ensure_npu()
    except RuntimeError as exc:
        raise RuntimeError(f"No CUDA or Ascend NPU accelerator is available. {exc}") from exc
    if backend.is_available():
        return torch.device("npu", rank)
    raise RuntimeError("No CUDA or Ascend NPU accelerator is available.")


def _backend(device):
    resolved = device_of(device)
    return resolved, None if resolved.type == "cpu" else getattr(torch, resolved.type)


def set_memory_fraction(fraction: float, device=None) -> None:
    """Set the CUDA allocator fraction; TorchNPU manages HBM allocation itself."""
    resolved = device_of(device)
    if resolved.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction, device=resolved)


def memory_allocated(device=None) -> int:
    resolved, backend = _backend(device)
    return backend.memory_allocated(resolved) if backend is not None else 0


def max_memory_allocated(device=None) -> int:
    resolved, backend = _backend(device)
    return backend.max_memory_allocated(resolved) if backend is not None else 0


def memory_reserved(device=None) -> int:
    resolved, backend = _backend(device)
    return backend.memory_reserved(resolved) if backend is not None else 0


def reset_peak_memory_stats(device=None) -> None:
    resolved, backend = _backend(device)
    if backend is not None:
        backend.reset_peak_memory_stats(resolved)


def get_rng_state(device=None):
    resolved, backend = _backend(device)
    return backend.get_rng_state(resolved) if backend is not None else torch.get_rng_state()


def set_rng_state(state, device=None) -> None:
    resolved, backend = _backend(device)
    if backend is not None:
        backend.set_rng_state(state, resolved)
    else:
        torch.set_rng_state(state)


def synchronize(device=None) -> None:
    """Synchronize CUDA or Ascend work; CPU is already synchronous."""
    resolved, backend = _backend(device)
    if backend is not None:
        backend.synchronize(resolved)


def empty_cache(device=None) -> None:
    """Return cached allocator blocks to the active accelerator driver."""
    _, backend = _backend(device)
    if backend is not None:
        backend.empty_cache()


def set_device(device) -> None:
    """Bind the current process to a CUDA or Ascend device."""
    resolved, backend = _backend(device)
    if backend is not None:
        backend.set_device(resolved)


def generator(device, seed: int) -> torch.Generator:
    """Create a seeded generator on the requested device."""
    return torch.Generator(device=str(device_of(device))).manual_seed(seed)


@contextlib.contextmanager
def autocast(device, *, dtype=torch.float16, enabled: bool = True) -> Iterator[None]:
    """Autocast on the requested device, with a CPU-safe no-op fallback."""
    resolved = device_of(device)
    if not enabled or resolved.type == "cpu":
        yield
        return
    with torch.autocast(device_type=resolved.type, dtype=dtype):
        yield
