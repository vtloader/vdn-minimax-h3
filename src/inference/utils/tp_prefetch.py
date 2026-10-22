"""Bounded one-block-ahead prefetch for the retained TP offload manager.

This module is intentionally separate from ``tp_offload.py``.  It consumes the
manager created there after all CPU backing has been constructed, and changes
only that manager instance's group callbacks and reset/snapshot methods.

The copy stream performs H2D transfers only.  HCCL and model execution remain on
the current stream and in Diffusers' existing group order.  At most the current
and next block leases are retained on the NPU.  CPU backing is immutable for the
life of the manager.
"""
from __future__ import annotations

import atexit
import copy
import time
import weakref
from types import MethodType

import torch
import torch.distributed as dist

from src.inference.utils.npu_pinned_offload import read_mem_available_bytes


GIB = 1024 ** 3
HOST_RESERVE_FACTOR = 1.5
HOST_HEADROOM_BYTES = 8 * GIB
MAX_DEVICE_LEASES = 2


def _npu_backend():
    backend = getattr(torch, "npu", None)
    if backend is None or not hasattr(backend, "Stream") or not hasattr(backend, "Event"):
        raise RuntimeError(
            "TP prefetch requires TorchNPU torch.npu.Stream and torch.npu.Event"
        )
    return backend


def _current_stream(backend, device):
    return backend.current_stream(device)


def _host_stats(backend):
    memory = getattr(backend, "memory", None)
    stats_fn = getattr(memory, "host_memory_stats", None)
    if stats_fn is None:
        return None
    try:
        stats = stats_fn()
    except Exception as exc:  # optional telemetry; pinning itself must still fail loudly
        return {"error": f"{type(exc).__name__}: {exc}"}
    keys = (
        "allocation.current", "allocation.peak", "reserved_bytes.current",
        "reserved_bytes.peak", "allocated_bytes.current", "allocated_bytes.peak",
        "num_host_alloc", "num_host_free", "host_alloc_time.total",
        "host_alloc_time.count",
    )
    return {key: stats[key] for key in keys if key in stats}


def _host_reserved_bytes(stats):
    if isinstance(stats, dict):
        value = stats.get("reserved_bytes.current")
        if isinstance(value, int):
            return value
    return None


def _logical_bytes(backing):
    return sum(source.numel() * source.element_size() for _, source in backing)


def _all_group_tensors(controller):
    # The retained manager already made this complete list.  Do not reconstruct
    # it from state_dict(): nonpersistent row_gather.full_weight is required.
    return [(tensor, source) for tensor, source in controller.backing]


class _Lease:
    def __init__(self, index, controller, copy_start, ready, entries, host_started):
        self.index = index
        self.controller = controller
        self.copy_start = copy_start
        self.ready = ready
        self.entries = entries
        self.host_started = host_started
        self.host_ready = time.perf_counter()
        self.done = None
        self.consumed = False
        self.bound = False
        self.restored = False
        self.device_bytes = sum(
            device_tensor.numel() * device_tensor.element_size()
            for _, _, device_tensor in entries
        )


class _PrefetchScheduler:
    def __init__(self, manager, transformer, root, blocks):
        self.manager = manager
        self.transformer = transformer
        self.backend = _npu_backend()
        self.device = root.device
        self.copy_stream = self.backend.Stream(device=self.device)
        self.root = root
        self.blocks = blocks
        self.block_count = len(blocks)
        self.leases = []
        self.current = None
        self.expected = 0
        self.completed_events = []
        self.runtime = {
            "h2d_bytes": 0,
            "onload_calls": 0,
            "offload_calls": 0,
            "prefetch_calls": 0,
            "blocks_consumed": 0,
            "retirement_host_waits": 0,
            "retirement_host_wait_seconds": 0.0,
            "restore_host_seconds": 0.0,
            "consume_host_seconds": 0.0,
            "prefetch_launch_host_seconds": 0.0,
            "max_outstanding_leases": 0,
            "max_device_lease_bytes": 0,
            "pending_leases": 0,
        }
        self.setup = {}
        self._closed = False
        self._original_backing = {
            id(controller): list(controller.backing)
            for controller in manager.groups
        }
        self._pin_backing_collectively()

    def _pin_backing_collectively(self):
        if self.device.type != "npu":
            raise ValueError(f"TP prefetch requires an NPU device, got {self.device}")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("TP prefetch requires an initialized two-rank process group")
        world = dist.get_world_size()
        rank = dist.get_rank()
        if world != 2:
            raise RuntimeError(f"TP prefetch requires exactly two ranks, got world_size={world}")

        local_bytes = sum(_logical_bytes(controller.backing)
                          for controller in self.manager.groups)
        local_available = read_mem_available_bytes()
        reports = [None, None]
        dist.all_gather_object(
            reports,
            {"rank": rank, "backing_bytes": local_bytes,
             "available_bytes": local_available},
        )
        if any(report is None for report in reports):
            raise RuntimeError("TP prefetch host guard received an incomplete rank report")
        if any(report["available_bytes"] is None for report in reports):
            raise RuntimeError("TP prefetch cannot read MemAvailable on every rank")

        total_bytes = sum(int(report["backing_bytes"]) for report in reports)
        min_available = min(int(report["available_bytes"]) for report in reports)
        required = int(HOST_RESERVE_FACTOR * total_bytes) + HOST_HEADROOM_BYTES
        guard = {
            "world_size": world,
            "rank_reports": reports,
            "total_backing_bytes": total_bytes,
            "total_backing_gib": total_bytes / GIB,
            "min_available_bytes": min_available,
            "min_available_gib": min_available / GIB,
            "required_bytes": required,
            "required_gib": required / GIB,
            "reserve_formula": "1.5x sum(backing bytes across both ranks) + 8 GiB",
        }
        if min_available < required:
            raise RuntimeError(
                "TP prefetch host guard failed: both-rank MemAvailable minimum "
                f"{min_available / GIB:.1f} GiB is below required {required / GIB:.1f} GiB"
            )

        before = _host_stats(self.backend)
        started = time.perf_counter()
        # Both ranks pin their own 36 GiB backing at the same time. The host guard
        # above already required 1.5x the combined backing plus 8 GiB free, so the
        # two copies fit; pinning them concurrently halves the cold-start pin wall.
        dist.barrier()
        error = None
        try:
            self._pin_local_backing()
        except BaseException as exc:
            error = f"rank {rank}: {type(exc).__name__}: {exc}"
        pin_errors = [None, None]
        dist.all_gather_object(pin_errors, error)
        dist.barrier()
        if any(pin_errors):
            self._restore_cpu_bindings_after_setup_failure()
            raise RuntimeError("TP prefetch backing pin failed: "
                               + "; ".join(item for item in pin_errors if item))

        after = _host_stats(self.backend)
        self.setup = {
            "pin_seconds": time.perf_counter() - started,
            "logical_pinned_host_bytes": local_bytes,
            "host_guard": guard,
            "host_allocator_before": before,
            "host_allocator_after": after,
            "setup_scope": "host wall interval; includes pin_memory copies and rank serialization",
        }
        # Do not keep the pre-pin detached tensors alive: that would retain a
        # second pageable copy of the complete model for the manager lifetime.
        self._original_backing = None

    def _pin_local_backing(self):
        try:
            for controller in self.manager.groups:
                original = list(controller.backing)
                pinned = []
                for tensor, source in original:
                    pinned_source = source if source.is_pinned() else source.pin_memory()
                    tensor.data = pinned_source
                    pinned.append((tensor, pinned_source))
                controller.backing = pinned
        except BaseException:
            for controller in self.manager.groups:
                original = self._original_backing[id(controller)]
                controller.backing = list(original)
                for tensor, source in original:
                    tensor.data = source
            raise

    def _restore_cpu_bindings_after_setup_failure(self):
        for controller in self.manager.groups:
            original = self._original_backing[id(controller)]
            controller.backing = list(original)
            for tensor, source in original:
                tensor.data = source

    def _record_group_h2d(self, controller, bytes_count):
        controller.metrics["h2d_bytes"] += bytes_count
        self.runtime["h2d_bytes"] += bytes_count

    def _launch(self, index):
        if index >= self.block_count:
            return None
        if len(self.leases) >= MAX_DEVICE_LEASES:
            self._retire_oldest()
        if len(self.leases) >= MAX_DEVICE_LEASES:
            raise RuntimeError("TP prefetch lease bound exceeded before admitting a block")

        controller = self.blocks[index]
        entries = []
        started = time.perf_counter()
        copy_start = self.backend.Event(enable_timing=True)
        ready = self.backend.Event(enable_timing=True)
        with self.backend.stream(self.copy_stream):
            copy_start.record(self.copy_stream)
            with torch.profiler.record_function(f"tp_prefetch::block_{index}::h2d"):
                for tensor, source in _all_group_tensors(controller):
                    device_tensor = source.to(self.device, non_blocking=True)
                    entries.append((tensor, source, device_tensor))
            ready.record(self.copy_stream)
        lease = _Lease(index, controller, copy_start, ready, entries, started)
        self.leases.append(lease)
        self.runtime["prefetch_calls"] += 1
        self.runtime["prefetch_launch_host_seconds"] += time.perf_counter() - started
        bytes_count = _logical_bytes(controller.backing)
        self._record_group_h2d(controller, bytes_count)
        self.runtime["max_outstanding_leases"] = max(
            self.runtime["max_outstanding_leases"], len(self.leases))
        self.runtime["max_device_lease_bytes"] = max(
            self.runtime["max_device_lease_bytes"],
            sum(item.device_bytes for item in self.leases),
        )
        return lease

    def start_forward(self):
        if self.leases or self.current is not None:
            raise RuntimeError("TP prefetch forward started with outstanding block leases")
        self.expected = 0
        self._launch(0)

    def consume(self, index):
        if index != self.expected:
            raise RuntimeError(
                f"TP prefetch block order changed: expected block {self.expected}, got {index}"
            )
        started = time.perf_counter()
        lease = next((item for item in self.leases if item.index == index), None)
        if lease is None or lease.consumed:
            raise RuntimeError(f"TP prefetch has no unconsumed lease for block {index}")
        compute = _current_stream(self.backend, self.device)
        # This is a stream dependency, not a host wait. The assignment below is
        # safe because all future current-stream work is ordered after ready.
        compute.wait_event(lease.ready)
        for tensor, source, device_tensor in lease.entries:
            if device_tensor.numel():
                device_tensor.record_stream(compute)
            tensor.data = device_tensor
        lease.consumed = True
        lease.bound = True
        self.current = lease
        self.expected += 1
        self.runtime["blocks_consumed"] += 1
        self.runtime["onload_calls"] += 1
        lease.controller.metrics["onload_calls"] += 1
        self.runtime["consume_host_seconds"] += time.perf_counter() - started
        # Admit next only after the previous block's post-forward callback has
        # recorded its done event. This keeps current + next bounded at two.
        self._launch(index + 1)
        return lease

    def finish_block(self, index):
        if self.current is None or self.current.index != index:
            raise RuntimeError(f"TP prefetch block {index} finished without its current lease")
        if not self.current.bound or self.current.restored:
            raise RuntimeError(f"TP prefetch block {index} has invalid lease state")
        lease = self.current
        compute = _current_stream(self.backend, self.device)
        lease.done = self.backend.Event(enable_timing=True)
        lease.done.record(compute)
        started = time.perf_counter()
        for tensor, source, _ in lease.entries:
            tensor.data = source
        lease.restored = True
        lease.controller.metrics["offload_calls"] += 1
        lease.controller.metrics["restore_host_seconds"] += time.perf_counter() - started
        self.runtime["offload_calls"] += 1
        self.runtime["restore_host_seconds"] += time.perf_counter() - started
        self.current = None
        self.runtime["pending_leases"] = len(self.leases)

    def _retire_oldest(self):
        if not self.leases:
            return
        lease = self.leases[0]
        if not lease.restored:
            raise RuntimeError(
                f"cannot retire block {lease.index} before its post-forward restoration"
            )
        started = time.perf_counter()
        if lease.done is None:
            raise RuntimeError(f"block {lease.index} has no completion event")
        if not lease.done.query():
            lease.done.synchronize()
            self.runtime["retirement_host_waits"] += 1
        waited = time.perf_counter() - started
        self.runtime["retirement_host_wait_seconds"] += waited
        self._complete_event_record(lease)
        self.leases.pop(0)
        self.runtime["pending_leases"] = len(self.leases)

    def _complete_event_record(self, lease):
        if lease.ready.query() is False:
            return
        record = {
            "block": lease.index,
            "copy_host_seconds": lease.host_ready - lease.host_started,
            "ready_to_consume_host_seconds": None,
            "copy_device_seconds": lease.copy_start.elapsed_time(lease.ready) / 1000.0,
            "ready_event_complete": True,
            "completion_event_complete": lease.done is not None and lease.done.query(),
        }
        if lease.done is not None and lease.done.query():
            record["ready_to_done_device_seconds"] = (
                lease.ready.elapsed_time(lease.done) / 1000.0)
        else:
            record["ready_to_done_device_seconds"] = None
        self.completed_events.append(record)

    def drain(self):
        """Synchronize pending copy/compute work and restore every block binding."""
        if not self.leases:
            self.runtime["pending_leases"] = 0
            return
        compute = _current_stream(self.backend, self.device)
        self.copy_stream.synchronize()
        compute.synchronize()
        for lease in list(self.leases):
            if not lease.restored:
                for tensor, source, _ in lease.entries:
                    tensor.data = source
                lease.restored = True
            if lease.done is None:
                lease.done = self.backend.Event(enable_timing=True)
                lease.done.record(compute)
            if not lease.done.query():
                lease.done.synchronize()
                self.runtime["retirement_host_waits"] += 1
            self._complete_event_record(lease)
        self.leases.clear()
        self.current = None
        self.runtime["pending_leases"] = 0

    def assert_cpu_bindings(self):
        misplaced = [
            (controller.label, tensor.device.type)
            for controller in self.manager.groups
            for tensor, _ in controller.backing
            if tensor.device.type != "cpu"
        ]
        if misplaced:
            raise RuntimeError(
                "TP prefetch left module parameters or buffers on the device: "
                f"{misplaced[:4]}"
            )

    def abort(self):
        """Failure cleanup; stream completion precedes every CPU rebinding."""
        self.copy_stream.synchronize()
        _current_stream(self.backend, self.device).synchronize()
        for controller in self.manager.groups:
            for tensor, source in controller.backing:
                tensor.data = source
        for lease in self.leases:
            lease.restored = True
        self.leases.clear()
        self.current = None
        self.runtime["pending_leases"] = 0

    def close(self):
        """Drain in-flight copies exactly once before the caller drops the model."""
        if self._closed:
            return None
        self._closed = True
        try:
            self.copy_stream.synchronize()
        except BaseException:
            pass
        try:
            self.drain()
        except BaseException:
            self.abort()
        return None

    def reset_metrics(self):
        if self.leases:
            raise RuntimeError("cannot reset TP prefetch metrics while leases are pending")
        for key in (
            "h2d_bytes", "onload_calls", "offload_calls", "prefetch_calls",
            "blocks_consumed", "retirement_host_waits", "retirement_host_wait_seconds",
            "restore_host_seconds", "consume_host_seconds", "prefetch_launch_host_seconds",
        ):
            self.runtime[key] = 0
        self.runtime["max_outstanding_leases"] = 0
        self.runtime["max_device_lease_bytes"] = 0
        self.runtime["pending_leases"] = 0
        self.completed_events.clear()

    def snapshot(self):
        pending = [lease.index for lease in self.leases]
        result = {
            "prefetch_strategy": "retained_one_block_lookahead",
            "prefetch_pinned_host_bytes": self.setup.get("logical_pinned_host_bytes", 0),
            "pin_seconds": self.setup.get("pin_seconds"),
            "host_allocator_reserved_bytes": _host_reserved_bytes(
                self.setup.get("host_allocator_after")),
            "host_guard": copy.deepcopy(self.setup.get("host_guard")),
            "host_allocator_before": copy.deepcopy(self.setup.get("host_allocator_before")),
            "host_allocator_after": copy.deepcopy(self.setup.get("host_allocator_after")),
            "max_outstanding_leases": self.runtime["max_outstanding_leases"],
            "max_device_lease_bytes": self.runtime["max_device_lease_bytes"],
            "pending_leases": len(pending),
            "pending_lease_indices": pending,
            "retirement_host_waits": self.runtime["retirement_host_waits"],
            "retirement_host_wait_seconds": self.runtime["retirement_host_wait_seconds"],
            "h2d_logical_bytes": self.runtime["h2d_bytes"],
            "d2h_logical_bytes": 0,
            "event_timings": list(self.completed_events),
            "prefetch_calls": self.runtime["prefetch_calls"],
            "blocks_consumed": self.runtime["blocks_consumed"],
            "prefetch_launch_host_seconds": self.runtime["prefetch_launch_host_seconds"],
            "consume_host_seconds": self.runtime["consume_host_seconds"],
            "restore_host_seconds": self.runtime["restore_host_seconds"],
            "timing_scope": {
                "host": "Python callback intervals, pin interval and explicit retirement waits",
                "device": "NPU event elapsed intervals after their events completed",
                "h2d": "logical tensor bytes; DMA executes on the private copy stream",
                "d2h": "zero; CPU backing is restored by reference",
            },
        }
        return result


def _validate_topology(manager, transformer):
    if getattr(manager, "strategy", None) != "retained":
        raise ValueError("TP prefetch requires the retained TP offload strategy")
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if not blocks:
        raise ValueError("TP prefetch requires transformer.transformer_blocks")
    controllers = list(getattr(manager, "groups", ()))
    if len(controllers) != len(blocks) + 1:
        raise ValueError(
            "TP prefetch requires exactly one retained root group and one group per transformer block"
        )
    root = [controller for controller in controllers
            if controller.group.onload_leader is transformer]
    if len(root) != 1:
        raise ValueError("TP prefetch requires exactly one retained transformer root group")
    by_block = []
    for index, block in enumerate(blocks):
        matches = [controller for controller in controllers
                   if controller.group.onload_leader is block]
        if len(matches) != 1:
            raise ValueError(
                f"TP prefetch requires one retained group with transformer_blocks[{index}] as onload leader"
            )
        by_block.append(matches[0])
    if set(id(controller) for controller in controllers) != (
            {id(root[0])} | {id(controller) for controller in by_block}):
        raise ValueError("TP prefetch found an unsupported retained group topology")
    for controller in controllers:
        if controller.device.type != "npu" or controller.strategy != "retained":
            raise ValueError("TP prefetch requires NPU retained controllers")
        onload = getattr(controller.group, "_onload_from_memory", None)
        offload = getattr(controller.group, "_offload_to_memory", None)
        if getattr(onload, "__self__", None) is not controller:
            raise ValueError("TP prefetch requires manager-bound retained onload callbacks")
        if getattr(offload, "__self__", None) is not controller:
            raise ValueError("TP prefetch requires manager-bound retained offload callbacks")
    return root[0], by_block


def enable_tp_prefetch(manager, transformer):
    """Enable retained one-block lookahead on one TP manager instance.

    The caller must invoke this after ``apply_tp_offload(..., strategy="retained")``
    and before the first forward.  All ranks must call it in the same order.  The
    setup performs HCCL object-gathers/barriers for the host-memory guard and pins
    one rank at a time.  No forward collective is introduced.
    """
    existing = getattr(manager, "_tp_prefetch_scheduler", None)
    if existing is not None:
        if existing.transformer is not transformer:
            raise ValueError("TP prefetch manager is already attached to another transformer")
        return manager

    root, blocks = _validate_topology(manager, transformer)
    scheduler = _PrefetchScheduler(manager, transformer, root, blocks)

    original_root_onload = root.group._onload_from_memory
    original_root_offload = root.group._offload_to_memory
    original_transformer_forward = transformer.forward
    original_reset = manager.reset_metrics
    original_snapshot = manager.snapshot

    def root_onload():
        original_root_onload()
        scheduler.start_forward()

    def root_offload():
        # The retained root callback synchronizes the current compute stream and
        # restores root references. Only then retire block leases and copy events.
        original_root_offload()
        scheduler.drain()
        scheduler.assert_cpu_bindings()

    root.group._onload_from_memory = root_onload
    root.group._offload_to_memory = root_offload

    for index, controller in enumerate(blocks):
        original_offload = controller.group._offload_to_memory

        def block_onload(index=index):
            scheduler.consume(index)

        def block_offload(index=index, original_offload=original_offload):
            # Do not call Diffusers' original offload: it performs D2H copies.
            # Keep the local variable to make accidental callback replacement
            # visible during review and to preserve the per-instance closure.
            del original_offload
            scheduler.finish_block(index)

        controller.group._onload_from_memory = block_onload
        controller.group._offload_to_memory = block_offload

    def guarded_forward(self, *args, **kwargs):
        try:
            return original_transformer_forward(*args, **kwargs)
        except BaseException:
            scheduler.abort()
            raise

    def reset_metrics(self):
        scheduler.reset_metrics()
        original_reset()

    def snapshot(self):
        base = original_snapshot()
        prefetch = scheduler.snapshot()
        if prefetch["max_outstanding_leases"] > MAX_DEVICE_LEASES:
            raise RuntimeError("TP prefetch lease metric exceeded the two-lease bound")
        # Keep existing manager fields intact while exposing scalar prefetch
        # fields at the same level for callers that consume the old snapshot.
        base.update({
            "strategy": "retained_prefetch",
            "base_strategy": base.get("strategy"),
            "prefetch": prefetch,
            "prefetch_pinned_host_bytes": prefetch["prefetch_pinned_host_bytes"],
            "pin_seconds": prefetch["pin_seconds"],
            "prefetch_host_allocator_reserved_bytes": prefetch[
                "host_allocator_reserved_bytes"],
            "max_outstanding_leases": prefetch["max_outstanding_leases"],
            "max_device_lease_bytes": prefetch["max_device_lease_bytes"],
            "pending_leases": prefetch["pending_leases"],
            "retirement_host_waits": prefetch["retirement_host_waits"],
            "event_timings": prefetch["event_timings"],
            "prefetch_timing_scope": prefetch["timing_scope"],
        })
        return base

    manager._tp_prefetch_scheduler = scheduler
    manager._tp_prefetch_original_reset = original_reset
    manager._tp_prefetch_original_snapshot = original_snapshot
    manager.reset_metrics = MethodType(reset_metrics, manager)
    manager.snapshot = MethodType(snapshot, manager)
    manager.drain = MethodType(lambda self: scheduler.drain(), manager)
    manager.abort = MethodType(lambda self: scheduler.abort(), manager)
    manager.close_prefetch = MethodType(lambda self: scheduler.close(), manager)
    transformer.forward = MethodType(guarded_forward, transformer)

    scheduler_ref = weakref.ref(scheduler)

    def close_at_exit():
        active = scheduler_ref()
        if active is not None:
            try:
                active.close()
            except BaseException:
                pass

    atexit.register(close_at_exit)
    return manager


__all__ = ["enable_tp_prefetch"]
