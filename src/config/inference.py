"""Inference overlay config. Four families, four sections: runtime optimization
(kernels), precision modification (fp8/dtype), multi-GPU layout (parallel, honoured by
infer_ulysses.py only; infer.py refuses non-default values), semantic ablation -- and
the last is gated: any semantic override with ablation.enabled false is a load-time
error, not a warning.

Every knob is a field here (YAML + dotlist); nothing on the inference path reads an
environment variable (FA_CLC is flash-attn's own). Arch-dependent choices resolve at
the setter, not the config: `softmax_backend: auto` is the decomposed window kernel on
any CUDA device, fp8 scale granularity follows the capability -- one YAML for both
architectures."""
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from src.config.common import KernelsConfig
from src.models.softmax_attention.decomposed import SOFTMAX_BACKENDS
from src.inference.geometry import (DEFAULT_HEIGHT, DEFAULT_WIDTH, EXPORT_FPS,
                                    validate_canvas)


@dataclass
class InferenceKernels(KernelsConfig):
    # softmax_backend (inherited): auto | flex | decomposed | ref | npu_varlen -- WHICH
    # window-softmax kernel. auto = decomposed on every CUDA device and ref on NPU
    # (hybrid_transform.set_softmax_backend); npu_varlen is NPU-only and explicit.
    #
    # inference_kernels: set_inference_mode(model, True) -- the forward-only kernel set,
    # one switch, same arithmetic:
    #   attention   fused QK-norm + RoPE in one pass, q/k/v consumed as views of the
    #               packed [T,H,d] buffer; the window softmax on the static
    #               FLASH/CuteDSL flex variant instead of the dynamic Triton one; the
    #               softmax gate fused with the to_out repack
    #   far branch  one-kernel frame-statistics prologue; Triton 5-tap temporal conv;
    #               compiled RMSNorm + output gate epilogue; one triangular solve for
    #               the vdn inverse; readout in the bmm-native layout; cached gather
    #               indices; no anchor zero-fill; baddbmm scan updates; TF32 on the one
    #               fp32 GEMM
    #   block       RMSNorm + AdaLN affine and gate + residual as two compiled kernels;
    #               SwiGLU as one compiled kernel
    # False = the released arithmetic spelled eagerly (rung 0 of the yaml ladder).
    inference_kernels: bool = True

    # NPU-only, in conjunction with render.offload_dit: keep the FIRST n DiT blocks
    # resident in HBM and stream the remaining blocks from CPU through Diffusers'
    # block-level group offloading. 0 = every block offloaded (the old NPU behaviour).
    # Requires a hybrid checkpoint; the last block must stay offloaded, because the
    # full 65.7 GiB static transformer does not fit a 64 GiB NPU.
    npu_resident_blocks: int = 0

    # NPU-only, opt-in, in conjunction with render.offload_dit: pin every CPU weight
    # once (about 97 GiB of host pinned memory for the 65.7 GiB transformer) and stream
    # blocks asynchronously through an NPU side stream. 39f 1NFE is 3.85x faster and
    # bitwise identical; 345f 1NFE is 1.31x faster. Requires a hybrid checkpoint and is
    # mutually exclusive with npu_resident_blocks.
    npu_pinned_offload: bool = False


@dataclass
class Fp8Config:
    enabled: bool = False                # NEVER default-on: fp8 changes the sample
    skip_end_blocks: int = 4             # scale granularity is per arch, not a knob


@dataclass
class PrecisionConfig:
    dtype: str = "bfloat16"
    fp8: Fp8Config = field(default_factory=Fp8Config)


@dataclass
class ParallelConfig:
    """torchrun layout for infer_ulysses.py."""
    softmax_ranks: int = 6               # 0 = standard Ulysses; n = n softmax + (world-n) linear
    profile: bool = False                # per-section CUDA events: profiling only
    # NPU-only, opt-in: true = standard head-parallel Ulysses (HCCL all-to-all QKV,
    # head-sharded softmax + linear branch); false = the sequence fallback that gathers
    # the residual and runs every block on every rank. Requires heads % WORLD_SIZE == 0
    # on 2+ NPUs; a bad value or a non-divisible head count is a hard error. Ignored on
    # CUDA (branch-parallel layout is selected there by softmax_ranks).
    npu_head_parallel: bool = False
    # Experimental weight sharding, dedicated infer_tp.py entrypoint, exactly 2 NPUs.
    # render.offload_dit also applies to these shards (synchronous block offload).
    npu_tensor_parallel: bool = False
    # Only honoured with npu_tensor_parallel=true. off = current split-K/partial TP
    # arithmetic; row_gather = complete-N padded Column + complete-K Row with the full
    # Row weight resident; row_stream = the same arithmetic but Row weights are gathered
    # for every forward. Both parity modes match the single-card BF16 GEMM shapes.
    npu_tp_parity: str = "off"
    # With TP + render.offload_dit: retained keeps immutable CPU backing and restores
    # references after compute; synchronous is the original copy-back baseline.
    tp_offload_strategy: str = "retained"
    # Precomputed merged rank shard cache (src/inference/prepare_tp_cache.py), only with
    # npu_tensor_parallel=true. Empty = assemble and merge on CPU at launch.
    tp_cache: str = ""
    # Where the checkpoint's LoRA pairs and external adapters are merged. "cpu" is the
    # existing assemble-on-CPU path; "npu" reads only this rank's base/branch shard and
    # merges it on the device (src/inference/utils/base_shard.py). The NPU merge is not
    # bitwise against CPU BF16 even in parity modes; it cannot use a precomputed cache.
    merge_device: str = "cpu"
    # With merge_device="npu": "npu" computes B @ A on the NPU (fast, a few ULP
    # non-bitwise vs CPU), "cpu" computes the FP32 delta exactly as the CPU merge and
    # applies the BF16 update on the NPU (slower, bitwise in the measured off run).
    merge_delta_device: str = "npu"


@dataclass
class BehaviorConfig:
    teacher_mode: bool = False


@dataclass
class AblationConfig:
    enabled: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderConfig:
    prompt_file: str = MISSING
    out: str = MISSING
    num_frames: int = 345
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = 24                         # export frame rate; model/audio stay at native 24 FPS
    num_steps: int = 50
    warmup_steps: int = 0                # NFE run and discarded first: 2 for any timing
                                         # (compile + the 2nd-timestep re-specialisation)
    seed: int = 42
    device: str = "cuda:0"
    offload_dit: bool = False             # block-level CPU offload, useful on 64 GB NPU
    video_shift: float = 12.0            # scheduler shifts; a distilled adapter is only
    audio_shift: float = 3.0             # valid at the shifts it was trained for
    save_latents: bool = False           # optional parity/debug tensors live outside
    latents_root: str = "artifacts/latents"  # results/: JSON/JSONL + MP4 only
    record: bool = False                 # also write <out>.inference.json (checkpoint
                                         # identity, resolved config, actual kernel
                                         # state, per-NFE timings) -- a debug aid


@dataclass
class ExternalLora:
    """A community adapter (safetensors, peft names against the DENSE model), merged
    after the checkpoint's own LoRA. alpha=None reads the file's metadata (missing
    there too -> alpha=rank, scale 1.0)."""
    path: str = MISSING
    alpha: Optional[float] = None


@dataclass
class InferenceConfig:
    checkpoint: Optional[str] = None     # None = render the DENSE base model
    base_source: Optional[str] = None    # base weights path override; identity stays with the spec
    vae_source: Optional[str] = None     # the decoders' root; None = the release copy (paths.H3_BASE)
    external_loras: List[ExternalLora] = field(default_factory=list)
    render: RenderConfig = field(default_factory=RenderConfig)
    kernels: InferenceKernels = field(default_factory=InferenceKernels)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


def validate_ablation(cfg) -> None:
    if cfg.ablation.overrides and not cfg.ablation.enabled:
        raise ValueError(
            f"semantic overrides {sorted(cfg.ablation.overrides)} require "
            f"ablation.enabled=true -- they change what the model computes"
        )


def validate_kernels(cfg) -> None:
    validate_canvas(cfg.render.width, cfg.render.height)
    if cfg.render.fps not in EXPORT_FPS:
        raise ValueError(f"render.fps must be one of {EXPORT_FPS} (export rate; native rate is 24)")
    if cfg.kernels.softmax_backend not in SOFTMAX_BACKENDS:
        raise ValueError(f"kernels.softmax_backend {cfg.kernels.softmax_backend!r} "
                         f"not in {SOFTMAX_BACKENDS}")
    if cfg.render.warmup_steps < 0:
        raise ValueError("render.warmup_steps must be >= 0")
    if cfg.precision.fp8.skip_end_blocks < 0:
        raise ValueError("precision.fp8.skip_end_blocks must be >= 0")
    if cfg.render.num_steps < 1:
        raise ValueError("render.num_steps must be >= 1")
    if cfg.render.num_frames <= 5:
        raise ValueError("render.num_frames must be > 5 (22 is the first aligned frame count)")
    if cfg.precision.fp8.enabled and cfg.render.device.split(":", 1)[0] != "cuda":
        raise ValueError("precision.fp8.enabled requires a CUDA device; use bf16 on NPU")
    if cfg.precision.fp8.enabled and cfg.render.offload_dit:
        raise ValueError("render.offload_dit with the repository's Triton FP8 path is "
                         "unsupported; use bf16 offload or CUDA resident FP8")
    if cfg.kernels.npu_resident_blocks < 0:
        raise ValueError("kernels.npu_resident_blocks must be >= 0")
    if cfg.kernels.npu_resident_blocks:
        if cfg.render.device.split(":", 1)[0] != "npu":
            raise ValueError("kernels.npu_resident_blocks is NPU-only; use "
                             "render.device=npu:<index> or leave it at 0")
        if not cfg.render.offload_dit:
            raise ValueError("kernels.npu_resident_blocks requires render.offload_dit=true "
                             "(resident blocks are the part left off the offload hooks)")

    if getattr(cfg.kernels, "npu_pinned_offload", False):
        if cfg.render.device.split(":", 1)[0] != "npu":
            raise ValueError("kernels.npu_pinned_offload is NPU-only; use "
                             "render.device=npu:<index> or leave it false")
        if not cfg.render.offload_dit:
            raise ValueError("kernels.npu_pinned_offload requires render.offload_dit=true")
        if not getattr(cfg, "checkpoint", None):
            raise ValueError("kernels.npu_pinned_offload requires a hybrid checkpoint "
                             "(checkpoint=null is the dense released model)")
        if cfg.kernels.npu_resident_blocks:
            raise ValueError("kernels.npu_pinned_offload and kernels.npu_resident_blocks "
                             "cannot be combined; the pinned path offloads every block")


def validate_parallel(cfg) -> None:
    """infer_ulysses.py: the layout is checked before the process group exists; the
    world-size bound (softmax_ranks < WORLD_SIZE) is checked at install."""
    if cfg.parallel.softmax_ranks < 0:
        raise ValueError("parallel.softmax_ranks must be >= 0 (0 = standard Ulysses)")
    if not isinstance(cfg.parallel.npu_head_parallel, bool):
        raise ValueError("parallel.npu_head_parallel must be true or false")
    parity = getattr(cfg.parallel, "npu_tp_parity", "off")
    if parity not in ("off", "row_gather", "row_stream"):
        raise ValueError("parallel.npu_tp_parity must be off, row_gather or row_stream "
                         '(quote "off" in YAML/dotlists; unquoted off parses as false)')
    if parity != "off" and not getattr(cfg.parallel, "npu_tensor_parallel", False):
        raise ValueError("parallel.npu_tp_parity is only valid with npu_tensor_parallel=true")
    if cfg.parallel.tp_offload_strategy not in ("synchronous", "retained", "prefetch"):
        raise ValueError("parallel.tp_offload_strategy must be synchronous, retained or prefetch")
    tp_cache = getattr(cfg.parallel, "tp_cache", "")
    if tp_cache:
        if not isinstance(tp_cache, str):
            raise ValueError("parallel.tp_cache must be a directory path")
        if not getattr(cfg.parallel, "npu_tensor_parallel", False):
            raise ValueError("parallel.tp_cache requires npu_tensor_parallel=true")
        if cfg.parallel.npu_head_parallel:
            raise ValueError("parallel.tp_cache cannot be combined with npu_head_parallel")
        if os.path.exists(tp_cache) and not os.path.isdir(tp_cache):
            raise ValueError(f"parallel.tp_cache must be a directory, got {tp_cache!r}")
    merge_device = getattr(cfg.parallel, "merge_device", "cpu")
    if merge_device not in ("cpu", "npu"):
        raise ValueError("parallel.merge_device must be 'cpu' or 'npu'")
    merge_delta_device = getattr(cfg.parallel, "merge_delta_device", "npu")
    if merge_delta_device not in ("cpu", "npu"):
        raise ValueError("parallel.merge_delta_device must be 'cpu' or 'npu'")
    if merge_delta_device != "npu" and merge_device != "npu":
        raise ValueError("parallel.merge_delta_device is only valid with "
                         "parallel.merge_device='npu'")
    if merge_device == "npu":
        if not getattr(cfg.parallel, "npu_tensor_parallel", False):
            raise ValueError("parallel.merge_device='npu' requires npu_tensor_parallel=true")
        if tp_cache:
            raise ValueError("parallel.merge_device='npu' cannot be combined with "
                             "parallel.tp_cache; the cache is already merged")
    if getattr(cfg.parallel, "npu_tensor_parallel", False):
        if cfg.parallel.npu_head_parallel:
            raise ValueError("npu_tensor_parallel and npu_head_parallel are mutually exclusive")
        if cfg.render.device.split(":", 1)[0] != "npu":
            raise ValueError("npu_tensor_parallel is NPU-only")
        if (cfg.kernels.npu_pinned_offload or cfg.kernels.npu_resident_blocks
                or cfg.precision.fp8.enabled):
            raise ValueError("npu_tensor_parallel requires unquantized shards and "
                             "npu_resident_blocks=0, npu_pinned_offload=false; "
                             "use render.offload_dit for synchronous TP block offload")
        if cfg.behavior.teacher_mode or not cfg.checkpoint:
            raise ValueError("npu_tensor_parallel requires a hybrid checkpoint, no teacher_mode")
        if cfg.precision.dtype != "bfloat16":
            raise ValueError("npu_tensor_parallel currently requires precision.dtype=bfloat16")


def validate_single_process(cfg) -> None:
    """infer.py: a parallel.* override would silently do nothing on the single-GPU
    entrypoint, so it is refused."""
    if OmegaConf.to_container(cfg.parallel) != asdict(ParallelConfig()):
        raise ValueError("parallel.* requires src/inference/infer_ulysses.py or "
                         "src/inference/infer_tp.py (torchrun); infer.py is single-process")
