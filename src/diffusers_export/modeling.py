"""The VDN-H3 transformer as a plain diffusers component.

This is the LAST section of the remote code published under each checkpoint's
`diffusers/` directory on the Hub. `src/diffusers_export/export.py` concatenates the
`src.models` closure this file imports, in reading order, and appends what is below as
one `modeling_vdn_h3.py` -- one file, since diffusers' loader resolves multi-file remote
code reliably only from a Hub repo. Nothing here is imported by this repository's own
inference stack: `assemble.py` remains the one assembly path for `infer.py` and
`infer_ulysses.py`.

    AutoModel.from_pretrained("OpenVDN/vdn-minimax-h3",
                              subfolder="stage-dmd-step-250/diffusers",
                              trust_remote_code=True)

`from_pretrained` is overridden because the three things it assembles do not live in
one directory: the base transformer is 66 GB of MiniMax-H3 under `h3-base/`, and the
VDN weights are the linear branch and the LoRA adapters one level up from `diffusers/`.
Pointing at them instead of copying them is why the published layout holds no duplicate
weights. The paths are `config.json`'s `vdn` block, written relative to the checkpoint
directory that contains `diffusers/`.

Quantisation is torchao's, and it runs LAST. The base always loads in bf16: the LoRA
merge writes into `Linear.weight`, and a quantised weight cannot take that write.
Diffusers' own loader quantises each weight as it loads, before anything here runs, so
`quantization_config` is caught here rather than passed down, and only a `TorchAoConfig`
is accepted -- torchao is the backend that quantises a built model.

`fp8=True` is the recipe this repository renders with, in torchao's terms: every Linear
at least 4096 wide on both sides in fp8 e4m3, activations per row and weights per output
channel on sm90, both per tensor on sm100, fast accumulation. Through a pipeline it is
per-component:

    pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16,
                         fp8={"transformer": True})

A `quantization_config` of your own replaces that recipe; without a
`modules_to_not_convert` of its own it leaves the same narrow Linears alone. Either way
each quantised Linear then calls a torch.compile'd `F.linear`: torchao's eager path makes
several passes over the activation, one of them an fp32 copy of it, and compiled it is
one kernel. The weights are parameters, so diffusers' group offloading streams them as
they are.

The activation amax is floored (`ACTIVATION_AMAX_FLOOR`), for the preset and for a
Float8 dynamic-activation config of your own that sets no `activation_value_lb`. Rows of
exact zeros reach `to_out_linear`: the linear branch skips the anchor frames, so their
readout is zero, and torchao's per-row scale for a zero row is 0, which quantises the row
to NaN and, through attention, the whole output. Fp8Linear's kernels floor the scale at
1e-12; this is the same floor in torchao's terms, amax = scale * 448.
"""
import dataclasses
import json
import os
import posixpath
import types

import torch
import torch.nn.functional as F
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.quantizers import DiffusersAutoQuantizer
from diffusers.quantizers.quantization_config import QuantizationMethod

from src.inference.utils.lora import merge_lora_state
from src.models.hybrid_transform import (apply_hybrid_attention_transform, iter_hybrids,
                                         set_inference_mode, set_layout,
                                         set_softmax_backend)
from src.models.ops.fp8_linear import MIN_WIDTH, per_tensor_gemm
from src.models.sequence_layout import layout_from_indices

CONFIG_KEY = "vdn"
_COMPILED_LINEAR = {}


def _open(source, repo_path, **hub_kwargs):
    """`repo_path` inside a local directory or a Hub repo id, as a local file path."""
    if os.path.isdir(source):
        return os.path.join(source, *repo_path.split("/"))

    from huggingface_hub import hf_hub_download

    return hf_hub_download(source, repo_path,
                           **{k: v for k, v in hub_kwargs.items() if v is not None})


def _sibling(subfolder, relative):
    """A `vdn` block path -- relative to the checkpoint directory, one level above
    `diffusers/` -- as a path from the repository root."""
    return posixpath.normpath(posixpath.join(subfolder or ".", "..", relative))


def _load_branch(model, weights):
    """The transform's own tensors. Refuses a key with no parameter rather than
    dropping it: a branch that half-loads renders, badly."""
    params = dict(model.named_parameters())
    unknown = [k for k in weights if k not in params]
    if unknown:
        raise RuntimeError(f"{len(unknown)} branch keys have no parameter, e.g. "
                           f"{sorted(unknown)[:4]}")
    for name, value in weights.items():
        params[name].data.copy_(value.to(params[name].dtype))
    return len(weights)


def _wide(module):
    return (isinstance(module, torch.nn.Linear) and module.in_features >= MIN_WIDTH
            and module.out_features >= MIN_WIDTH)


def _check_quantization(config):
    """Before the 66 GB load: torchao is installed, and the config is one it applies."""
    import torchao  # noqa: F401  -- fp8 and quantization_config go through torchao
    if config is not None and getattr(config, "quant_method", None) != QuantizationMethod.TORCHAO:
        raise ValueError(f"{type(config).__name__} quantises while the weights load, before "
                         "the LoRA merge this component performs; pass a TorchAoConfig, "
                         "which is applied after it")


def _fp8_config():
    """`fp8=True` in torchao's terms. The scale granularity follows the card, as it does
    for Fp8Linear; fp8_linear's header says why sm100 is per tensor."""
    from diffusers import TorchAoConfig
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, PerRow,
                                      PerTensor)

    return TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(
        granularity=PerTensor() if per_tensor_gemm() else PerRow(),
        mm_config=Float8MMConfig(use_fast_accum=True), set_inductor_config=False,
        activation_value_lb=ACTIVATION_AMAX_FLOOR))


ACTIVATION_AMAX_FLOOR = 448.0 * 1e-12   # fp8 e4m3 max times Fp8Linear's scale floor


def _floored(quant):
    """The user's Float8 dynamic-activation config with the amax floor, if it sets none."""
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

    if (isinstance(quant, Float8DynamicActivationFloat8WeightConfig)
            and quant.activation_value_lb is None):
        return dataclasses.replace(quant, activation_value_lb=ACTIVATION_AMAX_FLOOR)
    return quant


def _linear(x, weight, bias):
    return F.linear(x, weight, bias)


def _compile_linears(model):
    """One compiled `F.linear` shared by every quantised Linear, shapes dynamic: the
    token streams and weight shapes would otherwise each be a recompile."""
    from torchao.utils import TorchAOBaseTensor

    if "fn" not in _COMPILED_LINEAR:
        config = torch._dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(config, name):
                setattr(config, name, max(getattr(config, name), 64))
        _COMPILED_LINEAR["fn"] = torch.compile(_linear, dynamic=True)
    compiled = _COMPILED_LINEAR["fn"]

    def forward(self, x):
        return compiled(x, self.weight, self.bias)

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and isinstance(module.weight, TorchAOBaseTensor):
            module.forward = types.MethodType(forward, module)
            count += 1
    return count


def _quantize(model, config):
    """torchao over the built, merged model. Diffusers' loader would skip
    `modules_to_not_convert` and `_keep_in_fp32_modules`; so does this, and a config
    that names no modules gets the narrow Linears, the ones `fp8=True` leaves in bf16."""
    from torchao.quantization import quantize_

    skip = config.modules_to_not_convert
    if skip is None:
        skip = [name for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not _wide(module)]
    skip = list(skip) + list(model._keep_in_fp32_modules or [])

    def convert(module, name):
        return (isinstance(module, torch.nn.Linear)
                and not any(name == key or f"{key}." in f"{name}." for key in skip))

    quantize_(model, _floored(config.get_apply_tensor_subclass()), filter_fn=convert)
    _compile_linears(model)
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.TORCHAO
    model.hf_quantizer = DiffusersAutoQuantizer.from_config(config)


def _layout(position_ids, video_indices, text_indices):
    """The packed sequence's shape for the hybrid layers, from the row indices the pipeline
    hands the transformer. The target video rows are the trailing contiguous run of
    `video_indices` -- keyframe conditioning rows, when there are any, come before the
    audio rows -- and their (h, w) rotary coordinates are the frame grid. `render.py`
    derives the same layout from `build_packed_sequence`'s own counts."""
    breaks = (video_indices[1:] != video_indices[:-1] + 1).nonzero()
    target = video_indices[int(breaks[-1]) + 1:] if breaks.numel() else video_indices
    grid = position_ids[target, 1:]
    frame_h, frame_w = torch.unique(grid[:, 0]).numel(), torch.unique(grid[:, 1]).numel()
    tokens_per_frame = frame_h * frame_w
    return layout_from_indices(target, target.numel() // tokens_per_frame, tokens_per_frame,
                               seq_len=position_ids.shape[0], frame_size=(frame_h, frame_w),
                               text_indices=text_indices)


class VDNMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    """MiniMax-H3 with the VDN hybrid attention transform applied and the checkpoint's
    LoRA adapters folded in. Same forward signature, same config and the same outputs as
    the base class, so every MiniMax-H3 pipeline block accepts it unchanged. The one
    thing forward adds is the packed layout: without one a hybrid layer runs full
    attention and skips its linear branch."""

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                timestep_indices, token_tags, position_ids, video_indices, audio_indices,
                text_indices, attention_kwargs=None, return_dict=True):
        # Spelled out rather than *args: the denoise block hands over only the layout
        # tensors that forward names.
        set_layout(self, _layout(position_ids, video_indices, text_indices))
        return super().forward(hidden_states, audio_hidden_states, encoder_hidden_states,
                               timestep, timestep_indices, token_tags, position_ids,
                               video_indices, audio_indices, text_indices,
                               attention_kwargs=attention_kwargs, return_dict=return_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, subfolder=None,
                        cache_dir=None, revision=None, token=None,
                        local_files_only=None, fp8=None, quantization_config=None,
                        softmax_backend=None, **kwargs):
        # softmax_backend: the window-softmax kernel, "flex" (the checkpoint's default) or
        # "decomposed" (the mask as dense calls: FA4's varlen kernel on sm90 and up,
        # torch's own on sm8x; faster on B200).
        from safetensors.torch import load_file

        source = pretrained_model_name_or_path
        hub = {"cache_dir": cache_dir, "revision": revision, "token": token,
               "local_files_only": local_files_only}

        with open(_open(source, posixpath.join(subfolder or ".", "config.json"), **hub)) as f:
            spec = json.load(f)[CONFIG_KEY]

        if fp8 is None:
            fp8 = spec.get("fp8", False)
        if fp8 and quantization_config is not None:
            raise ValueError("fp8=True is a quantization_config of its own; pass one or "
                             "the other")
        if fp8 or quantization_config is not None:
            _check_quantization(quantization_config)

        # The base checkpoint is mixed precision and carries no `torch_dtype`, so a load
        # that names no dtype keeps every stored dtype -- a different model from the one
        # this checkpoint was distilled against, at twice the memory. bf16 is what
        # `render.load_models` asks for; `_keep_in_fp32_modules` still holds back the
        # time embedder. A caller that names a dtype gets that one.
        base = spec["base"]
        base_source = base["source"]
        if os.path.isdir(source) and os.path.isdir(os.path.join(source, base["subfolder"])):
            base_source = source
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = torch.bfloat16
        model = super().from_pretrained(
            base_source, subfolder=base["subfolder"], revision=base.get("revision"),
            cache_dir=cache_dir, token=token, local_files_only=local_files_only, **kwargs)

        apply_hybrid_attention_transform(model, spec["transform"])
        _load_branch(model, load_file(_open(source, _sibling(subfolder, spec["branch"]), **hub)))
        for adapter in spec["adapters"]:
            merge_lora_state(model, load_file(_open(source, _sibling(subfolder, adapter), **hub)))

        model.eval().requires_grad_(False)
        for attn in iter_hybrids(model):
            attn.teacher_mode = False
            for parameter in attn.parameters():
                if parameter.dtype == torch.float32:
                    parameter.data = parameter.data.to(torch.bfloat16)

        # The overlay `assemble.build_inference_model` applies for a render: forward-only
        # kernel bodies and the window-softmax backend. Off makes this slower, never
        # wrong, so a caller that means to build a graph can set inference_kernels false.
        if spec.get("inference_kernels", True):
            set_inference_mode(model, True)
            # Models load on CPU before a pipeline moves/offloads them. Each fused
            # forward dispatches on its input tensor, so moving to NPU stays eager and
            # moving to CUDA keeps the original optimized path.
            set_softmax_backend(
                model,
                softmax_backend or spec.get("softmax_backend", "flex"),
            )

        # Last, after the merge (see the header), and off unless asked.
        if fp8 or quantization_config is not None:
            _quantize(model, _fp8_config() if fp8 else quantization_config)
        return model
