"""The window softmax as a union of DENSE attentions -- no BlockMask, no sparse kernel.

The c1 mask is not arbitrary sparsity: every kept pair lies in one of a few dense
rectangles. Split the QUERY rows into groups whose kept KV set is identical, and each
group is a plain dense attention:

  dense-q   global (text/audio) rows, plus anchor-frame queries under rows/both ->
            one flash_attn_func call against the FULL kv (no copy).
  windows   remaining frames grouped by identical window bounds (== chunks) -> ONE
            flash_attn_varlen_func call over per-group gathered kv
            [globals + window frames + anchor frames].

Same math as the masked kernel -- each query's softmax spans exactly its kept set in
one pass -- so outputs differ from flex by bf16 reduction order only (a few 1e-3
relative, the same distance two flex backends show against each other). On sm100 the
dense calls are markedly faster than the block-sparse flex kernel at this mask density;
on sm90 the two run at the same speed.

Inference-only (config kernels.softmax_backend = decomposed, and what auto resolves to on
every CUDA device; hybrid_transform.set_softmax_backend writes the choice onto every
hybrid layer): training keeps the flex path. No fallback: a failure here raises rather
than silently downgrading to a slower kernel.

The window leg's varlen kernel is chosen once per process by the card (`varlen_kernel`):
FA4's CuTe kernel on sm90, sm100 and sm110, the cards FA4 has kernels for; torch's own
`varlen_attn` (its flash kernels, the FA2 lineage, sm80 and up) elsewhere, i.e. on sm8x
and on consumer Blackwell (sm120). Nothing is caught: a missing kernel raises with the
install to fix. The dense leg is torch SDPA: cuDNN on the FA4 cards, where it beats FA4
at this shape, SDPA's own choice elsewhere.
"""


import itertools

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from src.models.sequence_layout import SequenceLayout
from src.models.softmax_attention.window import has_fa4_kernels

_PLAN_CACHE = {}
_KERNEL_CACHE = {}
MAX_CACHED_PLANS = 4
SOFTMAX_BACKENDS = ("auto", "flex", "decomposed", "ref", "npu_varlen")


def _fa4_varlen():
    from flash_attn.cute.interface import flash_attn_varlen_func

    def fa4(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                     max_seqlen_q=max_q, max_seqlen_k=max_k,
                                     softmax_scale=scale)
        return out[0] if isinstance(out, tuple) else out

    fa4.name = "fa4"
    return fa4


def _torch_varlen():
    from torch.nn.attention.varlen import varlen_attn

    def torch_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        return varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale)

    torch_varlen.name = "torch"
    return torch_varlen


def varlen_kernel():
    """The varlen kernel the window leg runs on, resolved once per process: FA4's on the
    cards it has kernels for, torch's own elsewhere (see the header). An ImportError
    here is the install to fix: flash-attn-4 on those cards, torch >= 2.13 elsewhere."""
    if "varlen" not in _KERNEL_CACHE:
        _KERNEL_CACHE["varlen"] = _fa4_varlen() if has_fa4_kernels() else _torch_varlen()
    return _KERNEL_CACHE["varlen"]


def _dense_backends():
    """cuDNN SDPA on the FA4 cards, where it is faster than FA4 at the dense leg's shape;
    elsewhere SDPA picks among its own kernels."""
    if has_fa4_kernels():
        return [SDPBackend.CUDNN_ATTENTION]
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION]      # no MATH: it materialises the full score matrix


def resolve_softmax_backend(backend: str, device=None) -> str:
    """Config ``kernels.softmax_backend`` -> the implementation that will run. ``auto``
    is ``decomposed`` on every CUDA device: on sm100 it is the faster kernel, on sm90 it
    matches flex at the production width and is the same distance from the fp32
    reference (bf16 reduction order). So inference runs ONE window kernel on both
    architectures, the exact window with no mask machinery; flex stays what training
    uses. A CUDA device with no varlen kernel raises here, naming the install.
    ``flex`` / ``decomposed`` force either kernel; ``ref`` is the eager reference for
    parity/debug. ``npu_varlen`` is the NPU-only explicit opt-in (TND fusion attention)
    and is rejected on every other device. ``auto`` on NPU stays ``ref``."""
    if backend not in SOFTMAX_BACKENDS:
        raise ValueError(f"kernels.softmax_backend={backend!r}; expected one of "
                         f"{SOFTMAX_BACKENDS}")
    device_type = None if device is None else torch.device(device).type
    if backend == "npu_varlen":
        if device_type != "npu":
            raise ValueError(
                "kernels.softmax_backend='npu_varlen' is NPU-only; select it with "
                f"an npu target device, got {device!r}"
            )
        return "npu_varlen"
    if device_type is not None and device_type != "cuda":
        # FlexAttention, FA4 varlen and the decomposed CUDA path are backend-specific.
        # The eager reference has the same mask semantics and is the portable path for
        # Ascend NPU (and CPU tests).
        return "ref"
    if backend == "auto":
        if not torch.cuda.is_available():
            return "flex"
        backend = "decomposed"
    if backend == "decomposed":
        varlen_kernel()                   # raise now, with the install named, not mid-render
    return backend


class _Plan:
    __slots__ = ("dense_q", "win_q", "kv_gather", "cu_q", "cu_k", "cu_q_host",
                 "cu_k_host", "max_q", "max_k", "has_windows")

    def __init__(self, layout: SequenceLayout, bounds, anchor_frames, device):
        S = layout.seq_len
        F, TPF = layout.num_frames, layout.tokens_per_frame
        vs, ve = layout.video_start, layout.video_end
        anchor_set = {0, F - 1} if anchor_frames in ("columns", "rows", "both") else set()
        dense_row_frames = anchor_set if anchor_frames in ("rows", "both") else set()
        dense_col_frames = anchor_set if anchor_frames in ("columns", "both") else set()

        def frame_rows(f):
            return (vs + f * TPF, vs + (f + 1) * TPF)

        global_ranges = [r for r in ((0, vs), (ve, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            return torch.cat([torch.arange(a, b, device=device) for a, b in ranges])

        # dense-q rows: globals + anchor-row frames
        dense_ranges = merge(global_ranges + [frame_rows(f) for f in sorted(dense_row_frames)])
        self.dense_q = cat_ranges(dense_ranges) if dense_ranges else torch.empty(
            0, dtype=torch.long, device=device)

        # window groups: consecutive frames sharing identical bounds (== chunks)
        groups = []
        for f in range(F):
            if f in dense_row_frames:
                continue
            if groups and bounds[groups[-1][-1]] == bounds[f] and groups[-1][-1] == f - 1:
                groups[-1].append(f)
            else:
                groups.append([f])

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(max(lo, 0), min(hi + 1, F))) | dense_col_frames)
            q_r = merge([frame_rows(f) for f in frames])
            kv_r = merge(global_ranges + [frame_rows(f) for f in kv_frames])
            qi, ki = cat_ranges(q_r), cat_ranges(kv_r)
            q_idx.append(qi); kv_idx.append(ki)
            q_lens.append(len(qi)); k_lens.append(len(ki))

        self.has_windows = bool(groups)
        # Host cumulative ends, leading zero included: the NPU varlen backend reads
        # these instead of syncing an NPU tensor with .tolist() on every block.
        self.cu_q_host = (0, *itertools.accumulate(q_lens))
        self.cu_k_host = (0, *itertools.accumulate(k_lens))
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(
                device, torch.int32)
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(
                device, torch.int32)
            self.max_q, self.max_k = max(q_lens), max(k_lens)
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)

        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")


def _plan(layout, bounds, anchor_frames, device):
    key = (layout.seq_len, layout.video_start, layout.num_frames,
           layout.tokens_per_frame, tuple(bounds), anchor_frames, str(device))
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = _Plan(layout, bounds, anchor_frames, device)
        while len(_PLAN_CACHE) > MAX_CACHED_PLANS:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    return _PLAN_CACHE[key]


def _require_torch_npu():
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "kernels.softmax_backend='npu_varlen' requires torch_npu; install the "
            "Ascend PyTorch extension for this device"
        ) from exc
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError(
            "kernels.softmax_backend='npu_varlen' requires an available Ascend NPU"
        )
    return torch_npu


def window_softmax_npu_varlen(query, key, value, layout, bounds, scale,
                              anchor_frames="none"):
    """[T, H, d] q/k/v -> [T, H, d] with the same window mask as the reference, on
    Ascend NPU: global (and anchor-row) queries run plain SDPA against the full kv, and
    each window group runs one TND ``npu_fusion_attention`` call over its gathered kv.
    Groups are processed one at a time so the gather is bounded by the largest group,
    not the whole sequence. Exact softmax over each query's kept set, but bf16 reduction
    order differs from ``ref`` -- this is an explicit, non-bitwise opt-in."""
    if query.device.type != "npu":
        raise RuntimeError(
            "window_softmax_npu_varlen requires an NPU tensor; "
            f"got device {query.device}"
        )
    torch_npu = _require_torch_npu()

    plan = _plan(layout, bounds, anchor_frames, query.device)
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if len(plan.dense_q):
        qd = query[plan.dense_q]
        od = scaled_dot_product_attention(
            qd.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), scale=scale)
        out[plan.dense_q] = od[0].transpose(0, 1)
    if plan.has_windows:
        for group in range(len(plan.cu_q_host) - 1):
            q_start, q_end = plan.cu_q_host[group], plan.cu_q_host[group + 1]
            k_start, k_end = plan.cu_k_host[group], plan.cu_k_host[group + 1]
            q_index = plan.win_q[q_start:q_end]
            qw = query[q_index].contiguous()
            kw = key[plan.kv_gather[k_start:k_end]].contiguous()
            vw = value[plan.kv_gather[k_start:k_end]].contiguous()
            out[q_index] = torch_npu.npu_fusion_attention(
                qw, kw, vw, query.shape[1], "TND", scale=scale,
                actual_seq_qlen=[q_end - q_start],
                actual_seq_kvlen=[k_end - k_start],
                sparse_mode=0)[0]
    return out


def window_softmax_decomposed(query, key, value, layout, bounds, scale,
                              anchor_frames="none"):
    """[T, H, d] q/k/v -> [T, H, d], exactly the window+anchors+globals mask, as
    dense calls scatter-written straight into a contiguous output (no cat +
    inverse-permute pass; the row sets are disjoint and cover [0, T)). Strided
    q is fine (indexing copies); strided k/v are copied contiguous up front --
    FA4 mis-addresses slice-strided operands on sm100, and gathers from a strided
    source are slower anyway, so one copy serves both legs. The dense-q leg runs on
    SDPA (cuDNN on sm90 and up, faster than FA4 at this shape); the window leg on
    `varlen_kernel()`."""
    varlen = varlen_kernel()

    plan = _plan(layout, bounds, anchor_frames, query.device)
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if len(plan.dense_q):
        qd = query[plan.dense_q]
        with sdpa_kernel(_dense_backends()):
            od = scaled_dot_product_attention(
                qd.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0), scale=scale)
        out[plan.dense_q] = od[0].transpose(0, 1)
    if plan.has_windows:
        kw = key[plan.kv_gather]
        vw = value[plan.kv_gather]
        out[plan.win_q] = varlen(query[plan.win_q], kw, vw, plan.cu_q, plan.cu_k,
                                 plan.max_q, plan.max_k, scale)
    return out
