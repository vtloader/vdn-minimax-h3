"""Experimental inference-only weight TP; full residual sequence on every rank.

Only two ranks are supported. Install on CPU *after* all checkpoint/LoRA merges.
Every sliced parameter is cloned: no shard retains its full source storage.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from src.models.hybrid_transform import iter_hybrids
from src.utils.device import synchronize


REQUEST_TENSORS = (
    "hidden_states", "audio_hidden_states", "encoder_hidden_states", "timestep",
    "timestep_indices", "token_tags", "position_ids", "video_indices", "audio_indices",
    "text_indices",
)
PARITY_MODES = ("off", "row_gather", "row_stream")


def validate_tp_launch(environ=None):
    """Only one node, global/local ranks 0/1, with two local workers is supported."""
    env = os.environ if environ is None else environ
    try:
        world, local_world, rank, local = (int(env[name]) for name in
                                          ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK"))
        node, nodes = int(env.get("GROUP_RANK", "0")), int(env.get("GROUP_WORLD_SIZE", "1"))
    except (KeyError, ValueError) as exc:
        raise ValueError("TP requires integer torchrun WORLD_SIZE/LOCAL_WORLD_SIZE/RANK/LOCAL_RANK") from exc
    if world != 2 or local_world != 2 or node != 0 or nodes != 1:
        raise ValueError("TP requires exactly two ranks on a single node (--nproc_per_node=2)")
    if local not in (0, 1) or rank != local:
        raise ValueError("TP requires matching global/local ranks 0 and 1 (NPU local0/1 only)")
    return rank, local


def interval(size, rank, world_size):
    if world_size != 2 or rank not in range(world_size):
        raise ValueError("tensor parallel requires exactly two ranks")
    if size % world_size:
        raise ValueError(f"dimension {size} must be divisible by {world_size}")
    return slice(rank * (size // world_size), (rank + 1) * (size // world_size))


def storage_bytes(module):
    tensors = list(module.parameters()) + list(module.buffers())
    parameter_bytes = sum(t.numel() * t.element_size() for t in module.parameters())
    if any(t.device.type == "meta" for t in tensors):
        # Meta has no storage to deduplicate; the cache manifest reports real bytes.
        return {"parameter_bytes": parameter_bytes,
                "unique_storage_bytes": sum(t.numel() * t.element_size() for t in tensors)}
    stores = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
              for t in tensors}
    return {"parameter_bytes": parameter_bytes,
            "unique_storage_bytes": sum(stores.values())}


def storage_by_device(module):
    """Logical weight/buffer storage placement, not allocator or driver HBM usage."""
    stores = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        stores[key] = tensor.untyped_storage().nbytes()
    result = {}
    for (device, _), nbytes in stores.items():
        result[device] = result.get(device, 0) + nbytes
    return result


@dataclass
class TensorParallelRuntime:
    rank: int
    world_size: int
    device: torch.device
    parity: str = "off"
    communication_seconds: float = 0.0
    collectives: int = 0
    events: list = field(default_factory=list)
    _scratch: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.parity not in PARITY_MODES:
            raise ValueError(f"unknown TP parity mode {self.parity!r}; expected {PARITY_MODES}")

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        dist.barrier()

    def collective(self, fn):
        # Device events measure stream time without serialising every layer on host.
        if self.device.type == "npu":
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            self.events.append((start, end))
        else:
            start = time.perf_counter()
            fn()
            self.communication_seconds += time.perf_counter() - start
        self.collectives += 1

    def scratch(self, kind, shape, dtype, device):
        """Bounded, shape-keyed temporary reused by every layer with that shape."""
        key = (kind, tuple(shape), dtype, device.type, device.index)
        buf = self._scratch.get(key)
        if buf is None or buf.device != device:
            buf = torch.zeros(shape, dtype=dtype, device=device)
            self._scratch[key] = buf
        return buf

    def release_scratch(self):
        """Drop parity scratch before decoding; model buffers move with the model."""
        self._scratch.clear()

    def profile(self):
        synchronize(self.device)
        includes = ["RowLinear FP32 all_reduce SUM", "AdaLN output-feature all_gather"]
        if self.parity == "row_gather":
            includes = ["Row all-gather of local activations",
                        "AdaLN output-table all_gather", "Column padN (no collective)"]
        elif self.parity == "row_stream":
            includes = ["Row all-gather of local weights and activations",
                        "AdaLN output-table all_gather", "Column padN (no collective)"]
        return {"communication_seconds": self.communication_seconds + sum(
            a.elapsed_time(b) / 1000 for a, b in self.events),
                "collectives": self.collectives,
                "parity_scratch_bytes": sum(t.numel() * t.element_size()
                                            for t in self._scratch.values()),
                "communication_metric_coverage": {
                    "measurement": ("NPU stream event elapsed time, includes waits"
                                    if self.device.type == "npu" else "host wall-clock time"),
                    "includes": includes,
                    "excludes": ["request tensor broadcasts", "barriers", "metadata all_gather_object"],
                    "exclusive_network_time": False,
                }}


def _gather_last(tensor, runtime):
    local = tensor.contiguous()
    parts = [torch.empty_like(local) for _ in range(runtime.world_size)]
    runtime.collective(lambda: dist.all_gather(parts, local))
    return torch.cat(parts, dim=-1)


class ColumnLinear(nn.Module):
    """Output-channel shard; optionally gather in original feature order (AdaLN).

    In parity modes the local rows are zero-padded to the complete output width, a
    complete-N BF16 GEMM runs, and only the local rows are kept.  This reproduces the
    single-card GEMM shape without gathering the full weight from the peer.  AdaLN
    still gathers its small ordered output table, now from exact local rows.
    """
    def __init__(self, linear, rows, runtime, gather=False):
        super().__init__()
        self.weight = nn.Parameter(linear.weight[rows].clone(), requires_grad=False)
        self.bias = (nn.Parameter(linear.bias[rows].clone(), requires_grad=False)
                     if linear.bias is not None else None)
        self.in_features = linear.in_features
        self.out_features = self.weight.shape[0]
        self.global_out_features = linear.out_features
        self.rows = rows if isinstance(rows, slice) else rows.clone()
        self.runtime, self.gather = runtime, gather

    def _full(self, x):
        device, dtype = self.weight.device, self.weight.dtype
        full_weight = self.runtime.scratch(
            "column_weight", (self.global_out_features, self.in_features), dtype, device)
        full_weight.zero_()
        full_weight[self.rows] = self.weight
        if self.bias is not None:
            full_bias = self.runtime.scratch(
                "column_bias", (self.global_out_features,), self.bias.dtype, device)
            full_bias.zero_()
            full_bias[self.rows] = self.bias
        else:
            full_bias = None
        local = F.linear(x, full_weight, full_bias)[..., self.rows]
        if not self.gather:
            return local
        # AdaLN needs the complete ordered table. Each rank computes exact local rows
        # with the complete output width, then the tiny table is gathered as before.
        parts = [torch.empty_like(local) for _ in range(self.runtime.world_size)]
        self.runtime.collective(lambda: dist.all_gather(parts, local.contiguous()))
        return torch.cat(parts, dim=-1)

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError("TP column projections are inference-only; use torch.no_grad()")
        if self.runtime.parity != "off":
            return self._full(x)
        local = F.linear(x, self.weight, self.bias)
        if not self.gather:
            return local
        parts = [torch.empty_like(local) for _ in range(self.runtime.world_size)]
        self.runtime.collective(lambda: dist.all_gather(parts, local.contiguous()))
        return torch.cat(parts, dim=-1)


class RowLinear(nn.Module):
    """Input-channel shard; FP32 SUM, cast back, then add bias exactly once.

    Parity modes replace the split-K partial and SUM with a complete-K GEMM.
    ``row_gather`` keeps the complete weight resident; ``row_stream`` gathers it on
    every forward and keeps only a transient AllGather/concatenation buffer.
    """
    def __init__(self, linear, columns, runtime):
        super().__init__()
        self.weight = nn.Parameter(linear.weight[:, columns].clone(), requires_grad=False)
        self.bias = (nn.Parameter(linear.bias.clone(), requires_grad=False)
                     if linear.bias is not None else None)
        self.in_features = self.weight.shape[1]
        self.out_features = linear.out_features
        self.columns = columns
        self.runtime = runtime
        self.released_local_weight_bytes = 0
        self.register_buffer("full_weight", None, persistent=False)

    def prepare_full_weight(self):
        if self.runtime.parity != "row_gather":
            return 0
        if self.full_weight is None:
            parts = [torch.empty_like(self.weight) for _ in range(self.runtime.world_size)]
            self.runtime.collective(
                lambda: dist.all_gather(parts, self.weight.contiguous()))
            self.full_weight = torch.cat(parts, dim=1)
            # The local column shard is redundant once the complete weight is resident.
            # Releasing it keeps row_gather at +half a Row weight per rank, not +1.5x.
            self.released_local_weight_bytes = (
                self.weight.numel() * self.weight.element_size())
            self.weight = nn.Parameter(
                torch.empty(0, dtype=self.full_weight.dtype, device=self.full_weight.device),
                requires_grad=False)
            return self.full_weight.numel() * self.full_weight.element_size()
        return 0

    def forward(self, x):
        if torch.is_grad_enabled():
            raise RuntimeError("TP row projections are inference-only; use torch.no_grad()")
        if self.runtime.parity != "off":
            x_full = _gather_last(x, self.runtime)
            if self.runtime.parity == "row_gather":
                if self.full_weight is None:
                    raise RuntimeError("row_gather parity requires prepare_full_weight() after onload")
                weight = self.full_weight
            else:
                local = self.weight.contiguous()
                parts = [torch.empty_like(local) for _ in range(self.runtime.world_size)]
                self.runtime.collective(lambda: dist.all_gather(parts, local))
                weight = torch.cat(parts, dim=1)
            return F.linear(x_full, weight, self.bias)
        local = F.linear(x, self.weight)
        total = local.float()
        self.runtime.collective(lambda: dist.all_reduce(total))
        total = total.to(local.dtype)
        return total if self.bias is None else total + self.bias


def prepare_tensor_parallel(model):
    """Materialise parity-only buffers after the shards have reached the device."""
    parity = None
    full_row_weight_bytes = 0
    released_local_weight_bytes = 0
    for module in model.modules():
        if isinstance(module, RowLinear):
            parity = module.runtime.parity
            before = module.released_local_weight_bytes
            full_row_weight_bytes += module.prepare_full_weight()
            released_local_weight_bytes += module.released_local_weight_bytes - before
    return {"parity": parity or "off", "full_row_weight_bytes": full_row_weight_bytes,
            "released_local_row_weight_bytes": released_local_weight_bytes,
            "resident_row_weight_delta_bytes": full_row_weight_bytes - released_local_weight_bytes}


def install_tensor_parallel(model, runtime, allow_meta: bool = False):
    """Instance-only module replacements; token refiner and small shared layers replicate."""
    interval(2, runtime.rank, runtime.world_size)
    allowed = {"cpu", "meta"} if allow_meta else {"cpu"}
    if any(p.device.type not in allowed for p in model.parameters()):
        raise ValueError("install tensor parallel on CPU before moving shards to device")
    if getattr(model, "_weight_tensor_parallel", False):
        raise ValueError("tensor parallel is already installed")
    if model.training:
        raise ValueError("tensor parallel is inference-only; call model.eval() before installation")
    hybrids = list(iter_hybrids(model))
    if not hybrids or len(hybrids) != len(model.transformer_blocks):
        raise ValueError("tensor parallel requires hybrid attention in every block")
    # Validate the whole model before mutating any parameter.
    for block in model.transformer_blocks:
        attn = block.attn
        interval(attn.num_heads, runtime.rank, runtime.world_size)
        if attn.linear_attention.head_dim != attn.head_dim:
            raise ValueError("shared QKV requires equal softmax and linear head dimensions")
        if any(getattr(attn.orig, name).out_features != attn.num_heads * attn.head_dim
               for name in ("to_q", "to_k", "to_v")):
            raise ValueError("QKV output dimensions must match heads * head_dim")
        if block.ff.net[0].proj.out_features != 2 * block.ff.net[2].in_features:
            raise ValueError("FFN must store concatenated equal SwiGLU halves")
        interval(block.ff.net[2].in_features, runtime.rank, runtime.world_size)
        interval(block.adaln_proj.linear.out_features, runtime.rank, runtime.world_size)
        if attn.teacher_mode:
            raise ValueError("teacher_mode is unsupported by tensor parallel")
    before = storage_bytes(model)
    sharded_ids = set()
    block_layouts = []
    for block in model.transformer_blocks:
        a = block.attn
        heads = interval(a.num_heads, runtime.rank, runtime.world_size)
        channels = slice(heads.start * a.head_dim, heads.stop * a.head_dim)
        local_heads = heads.stop - heads.start
        block_layouts.append({"global_heads": a.num_heads, "local_heads": local_heads,
                              "head_start": heads.start, "head_stop": heads.stop,
                              "head_dim": a.head_dim,
                              "global_ffn_intermediate": block.ff.net[2].in_features,
                              "local_ffn_intermediate": block.ff.net[2].in_features // runtime.world_size,
                              "global_adaln_features": block.adaln_proj.linear.out_features,
                              "local_adaln_features": block.adaln_proj.linear.out_features // runtime.world_size})
        a.global_num_heads = a.num_heads
        a.num_heads = a.orig.heads = local_heads
        a.orig.inner_dim = local_heads * a.head_dim
        for name in ("to_q", "to_k", "to_v"):
            setattr(a.orig, name, ColumnLinear(getattr(a.orig, name), channels, runtime))
        a.orig.to_out[0] = RowLinear(a.orig.to_out[0], channels, runtime)
        a.to_out_linear = RowLinear(a.to_out_linear, channels, runtime)
        if a.enable_softmax_gate:
            a.softmax_gate.up = ColumnLinear(a.softmax_gate.up, heads, runtime)
            a.softmax_gate.num_heads = local_heads
        branch = a.linear_attention
        branch.num_heads = local_heads
        branch.beta_proj = ColumnLinear(branch.beta_proj, heads, runtime)
        branch.output_gate.up = ColumnLinear(branch.output_gate.up, channels, runtime)
        branch.output_gate.num_heads = local_heads
        branch.alpha.up = ColumnLinear(branch.alpha.up, channels, runtime)
        branch.alpha.num_heads = local_heads
        branch.alpha.dt_bias = nn.Parameter(branch.alpha.dt_bias[channels].clone(), requires_grad=False)
        branch.alpha.A_log = nn.Parameter(branch.alpha.A_log[heads].clone(), requires_grad=False)
        sharded_ids.update((id(branch.alpha.dt_bias), id(branch.alpha.A_log)))
        if branch.short_conv is not None:
            for proj in branch.short_conv.projs:
                for suffix in ("sp", "tm"):
                    conv = getattr(branch.short_conv, f"{proj}_{suffix}")
                    conv.weight = nn.Parameter(conv.weight[channels].clone(), requires_grad=False)
                    sharded_ids.add(id(conv.weight))
                    conv.in_channels = conv.out_channels = conv.groups = local_heads * a.head_dim
        intermediate = block.ff.net[2].in_features
        ff_slice = interval(intermediate, runtime.rank, runtime.world_size)
        # SwiGLU stores [a, gate], not interleaved pairs.
        rows = torch.cat((torch.arange(ff_slice.start, ff_slice.stop),
                          torch.arange(intermediate + ff_slice.start, intermediate + ff_slice.stop)))
        block.ff.net[0].proj = ColumnLinear(block.ff.net[0].proj, rows, runtime)
        block.ff.net[2] = RowLinear(block.ff.net[2], ff_slice, runtime)
        adaln = block.adaln_proj.linear
        block.adaln_proj.linear = ColumnLinear(
            adaln, interval(adaln.out_features, runtime.rank, runtime.world_size), runtime, gather=True)
    model._weight_tensor_parallel = True
    # Root guard and request synchronization are part of installation, so direct
    # callers cannot accidentally skip them. No original/full weights are captured.
    model.register_forward_pre_hook(broadcast_request, with_kwargs=True, prepend=True)
    for module in model.modules():
        if isinstance(module, (ColumnLinear, RowLinear)):
            sharded_ids.add(id(module.weight))
            if isinstance(module, ColumnLinear) and module.bias is not None:
                sharded_ids.add(id(module.bias))
    scopes = {"sharded": [], "replicated": []}
    for name, parameter in model.named_parameters():
        scopes["sharded" if id(parameter) in sharded_ids else "replicated"].append(name)
    after = storage_bytes(model)
    return {"before": before, "after": after,
            "sharded_parameter_bytes": before["parameter_bytes"] - after["parameter_bytes"],
            "replicated_parameter_bytes": 2 * after["parameter_bytes"] - before["parameter_bytes"],
            "parameter_scopes": scopes,
            "block_layouts": block_layouts,
            "reduction_semantics": {
                "local_partial": "F.linear output in compute dtype; BF16 partials round BEFORE FP32 cast",
                "local_weight_dtype": str(model.transformer_blocks[0].attn.orig.to_out[0].weight.dtype),
                "sum_dtype": "float32", "output_dtype": "local partial dtype",
                "bias": "added once after SUM and output cast", "forces_fp32_local_gemm": False,
            },
            "replicated_token_refiner": storage_bytes(model.token_refiner)}


def broadcast_request(module, args, kwargs):
    """Inference-only root guard and rank-zero ownership of all tensor inputs.

    Both ranks must provide the same request shapes/dtypes. Broadcast writes in-place
    into caller video/audio views, so subsequent scheduler updates see rank-zero noise.
    Only None/empty attention_kwargs is supported: checkpoint LoRAs are already merged.
    """
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("TP root is inference-only: model.eval() and torch.no_grad() are required")
    if args:
        raise ValueError("TP transformer requires keyword request inputs")
    attention_kwargs = kwargs.get("attention_kwargs")
    if attention_kwargs is not None and (not isinstance(attention_kwargs, dict) or attention_kwargs):
        raise ValueError("TP supports only attention_kwargs=None or {}; LoRAs must be merged before slicing")
    unknown = set(kwargs) - set(REQUEST_TENSORS) - {"attention_kwargs", "return_dict"}
    if unknown:
        raise ValueError(f"unsupported TP request kwargs: {sorted(unknown)}")
    if not isinstance(kwargs.get("return_dict", True), bool):
        raise ValueError("TP return_dict must be a bool")
    # Validate everything before any broadcast, and use a fixed collective order.
    for name in REQUEST_TENSORS:
        if not isinstance(kwargs.get(name), torch.Tensor):
            raise ValueError(f"TP request {name} must be a Tensor (nested/scalar inputs unsupported)")
    for name in REQUEST_TENSORS:
        dist.broadcast(kwargs[name], src=0)
    return args, kwargs
