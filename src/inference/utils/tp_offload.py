"""TP-local offload of immutable, post-merge parameters and registered buffers.

The retained strategy keeps authoritative CPU storage independently of module .data.
Device copies are disposable: after the current compute stream finishes, eviction
restores CPU references instead of copying unchanged weights back from the NPU.
Diffusers still owns group boundaries, input movement and forward wrapping.
"""
from __future__ import annotations

import time

import torch

from src.inference.utils.assemble import apply_transformer_offload
from src.utils.device import device_of


OFFLOAD_STRATEGIES = ("synchronous", "retained")
_COUNTERS = ("h2d_bytes", "d2h_bytes", "onload_calls", "offload_calls")
_TIMINGS = ("onload_host_seconds", "compute_wait_host_seconds", "restore_host_seconds")


def _group_tensors(group):
    # Do not retain this list in synchronous mode: Module.to() can replace buffers.
    # Keeping references to the replaced objects would keep old device storage live.
    tensors = list(group.parameters) + list(group.buffers)
    for module in group.modules:
        tensors.extend(module.parameters())
        tensors.extend(module.buffers())
    return list(dict.fromkeys(tensors))


class _GroupTransfer:
    def __init__(self, group, label, device, strategy):
        self.group, self.label = group, label
        self.device, self.strategy = device, strategy
        self.original_onload = group._onload_from_memory
        self.original_offload = group._offload_to_memory
        # detach() owns the original storage independently of subsequent .data
        # rebinding, including nonpersistent buffers omitted by state_dict().
        self.backing = ([(tensor, tensor.detach()) for tensor in _group_tensors(group)]
                        if strategy == "retained" else [])
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {key: 0 for key in _COUNTERS}
        self.metrics.update({key: 0.0 for key in _TIMINGS})

    def onload(self):
        started = time.perf_counter()
        if self.device.type != "cpu":
            self.metrics["h2d_bytes"] += sum(
                tensor.numel() * tensor.element_size()
                for tensor in _group_tensors(self.group) if tensor.device.type == "cpu")
        with torch.profiler.record_function(f"tp_offload::{self.label}::onload"):
            if self.strategy == "synchronous":
                self.original_onload()
            else:
                for tensor, source in self.backing:
                    tensor.data = source.to(self.device, non_blocking=False)
        self.metrics["onload_calls"] += 1
        self.metrics["onload_host_seconds"] += time.perf_counter() - started

    def offload(self):
        self.metrics["offload_calls"] += 1
        if self.strategy == "synchronous":
            self.metrics["d2h_bytes"] += sum(
                tensor.numel() * tensor.element_size()
                for tensor in _group_tensors(self.group) if tensor.device.type != "cpu")
            started = time.perf_counter()
            with torch.profiler.record_function(f"tp_offload::{self.label}::copy_back"):
                self.original_offload()
            self.metrics["restore_host_seconds"] += time.perf_counter() - started
            return

        started = time.perf_counter()
        with torch.profiler.record_function(f"tp_offload::{self.label}::wait_compute"):
            if self.device.type != "cpu":
                # HCCL's existing current-stream dependencies are unchanged. Do not
                # drop execution storage until that stream has consumed the weights.
                getattr(torch, self.device.type).current_stream(self.device).synchronize()
        self.metrics["compute_wait_host_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        with torch.profiler.record_function(f"tp_offload::{self.label}::restore_backing"):
            for tensor, source in self.backing:
                tensor.data = source
        self.metrics["restore_host_seconds"] += time.perf_counter() - started


class TPBlockOffload:
    def __init__(self, groups, strategy):
        self.groups, self.strategy = groups, strategy

    def reset_metrics(self):
        """Exclude setup/warmup transfers without replacing the CPU backing."""
        for group in self.groups:
            group.reset_metrics()

    def snapshot(self):
        """CPU-only metadata; timings are host intervals, not pure DMA duration."""
        by_group = {group.label: dict(group.metrics) for group in self.groups}
        totals = {key: sum(metrics[key] for metrics in by_group.values())
                  for key in (*_COUNTERS, *_TIMINGS)}
        return {"strategy": self.strategy,
                "host_backing_bytes": sum(source.numel() * source.element_size()
                                          for group in self.groups for _, source in group.backing),
                **totals, "by_group": by_group,
                "transfer_byte_semantics": "logical tensor bytes, excluding allocator padding",
                "timing_scope": ("host wrapper intervals; synchronous copy-back also waits for "
                                 "queued compute; not exclusive DMA/kernel timings")}


def apply_tp_offload(transformer, device, strategy="retained"):
    """Install on a frozen CPU model after merges, TP replacement and parity prep.

    All parameters AND registered buffers must be immutable during rendering. This
    includes row_gather.full_weight. The manager changes transfer ownership only;
    projection arithmetic and the collective sequence remain unchanged.
    """
    if strategy not in OFFLOAD_STRATEGIES:
        raise ValueError(f"unknown TP offload strategy {strategy!r}; expected {OFFLOAD_STRATEGIES}")
    # TP installs inference-only Linear wrappers after model.eval(); those newly
    # constructed wrappers have no train/eval-dependent operations of their own.
    if transformer.training:
        raise ValueError("TP offload requires model.eval()")
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise ValueError("TP offload requires frozen parameters")
    tensors = list(transformer.parameters()) + list(transformer.buffers())
    if any(tensor.device.type != "cpu" or tensor.requires_grad for tensor in tensors):
        raise ValueError("TP offload requires immutable CPU parameters and buffers")
    if any(hasattr(module, "_diffusers_hook") for module in transformer.modules()):
        raise ValueError("install TP offload before other offload hooks")
    device = device_of(device)
    apply_transformer_offload(transformer, device)
    groups, seen = [], set()
    for name, module in transformer.named_modules():
        registry = getattr(module, "_diffusers_hook", None)
        hook = registry.get_hook("group_offloading") if registry is not None else None
        if hook is None or id(hook.group) in seen:
            continue
        seen.add(id(hook.group))
        controller = _GroupTransfer(hook.group, name or "root", device, strategy)
        hook.group._onload_from_memory = controller.onload
        hook.group._offload_to_memory = controller.offload
        groups.append(controller)
    return TPBlockOffload(groups, strategy)
