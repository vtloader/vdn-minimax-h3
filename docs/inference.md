# Inference with this repository

How the repository's own stack renders: the entry points, the configs, the kernels behind them, fp8, the eight-GPU layout, memory, and what a prompt cache carries. The commands to run are in the [README](../README.md#inference-with-our-repository); the portable diffusers path is in [diffusers.md](diffusers.md).

## Entry points

| Script | What it does |
|---|---|
| [src/inference/infer.py](../src/inference/infer.py) | one GPU |
| [src/inference/infer_ulysses.py](../src/inference/infer_ulysses.py) | eight GPUs under `torchrun`, sequence parallel |
| [src/inference/encode_prompt.py](../src/inference/encode_prompt.py) | a text prompt to a prompt cache (`.pt`) |
| [src/inference/encode_keyframes.py](../src/inference/encode_keyframes.py) | a prompt plus keyframes (`--first`, `--last`), reference images (`--refs`), or ref2va videos/audio (`--video_refs`, `--audio_refs`) to a prompt cache |
| [src/inference/infer_diffusers.py](../src/inference/infer_diffusers.py) | the published diffusers component; shares no code with the two above |

Encoding is separate from rendering because the Qwen3-VL text encoder is 63 GB: a prompt is encoded once, and every render reuses the cache without loading the encoder.

## Configs

A config file says how to compute, and the command line says what: the checkpoint, the prompt cache, the output. Every knob is a field of [src/config/inference.py](../src/config/inference.py), set in the YAML or as a `key=value` override. Nothing on the inference path reads an environment variable.

The files in [configs/inference/](../configs/inference/) form a ladder, once for 8 and once for 50 model evaluations:

| File | Adds |
|---|---|
| `8nfe.yaml` | the released arithmetic in bf16; with `checkpoint=null`, the dense MiniMax-H3 |
| `8nfe_tuned.yaml` | the inference kernels; the same sample up to kernel rounding |
| `8nfe_tuned_fp8.yaml` | fp8 on every wide Linear, 2 warm-up evaluations; the top single-GPU tier |
| `8nfe_tuned_fp8_ulysses_h200.yaml`, `_b200.yaml` | the eight-GPU layout tuned for that card |

The fields:

| Field | Default | Meaning |
|---|---|---|
| `checkpoint` | `null` | a checkpoint directory; `null` renders the dense base model |
| `base_source`, `vae_source` | the release copy | where the base transformer and the two decoders load from |
| `external_loras` | `[]` | community adapters, see [Weights](#weights) |
| `kernels.inference_kernels` | `true` | the forward-only kernel set |
| `kernels.softmax_backend` | `auto` | `auto`, `flex`, `decomposed`, `ref` or `npu_varlen` (NPU-only) |
| `precision.fp8.enabled` | `false` | fp8 on the wide Linears |
| `precision.fp8.skip_end_blocks` | `4` | first and last N blocks left in bf16; the shipped fp8 configs use 0 |
| `parallel.softmax_ranks` | `6` | `infer_ulysses.py` only: ranks given to the softmax heads; 0 is standard Ulysses |
| `parallel.npu_head_parallel` | `false` | NPU-only experimental opt-in: use the standard head-parallel Ulysses path instead of the sequence-gather fallback; needs 2+ ranks and `heads % WORLD_SIZE == 0` |
| `parallel.npu_tensor_parallel` | `false` | experimental two-NPU weight TP, dedicated `infer_tp.py` only; exactly two local ranks and BF16 |
| `parallel.npu_tp_parity` | `"off"` | with `npu_tensor_parallel=true`: `"off"`, `row_gather`, or `row_stream`; the latter two reproduce single-card BF16 GEMM shapes. Frame/canvas limits are experimental warnings only, not validator gates |
| `parallel.tp_offload_strategy` | `"retained"` | with TP and `render.offload_dit=true`: `retained` keeps immutable CPU backing and avoids D2H weight copies; `synchronous` is the copy-back baseline |
| `parallel.tp_cache` | `""` | precomputed merged rank shard cache, only with `npu_tensor_parallel=true`; see [the cache section](#precomputed-merged-shard-cache-paralleltp_cache) |
| `parallel.merge_device` | `"cpu"` | with `npu_tensor_parallel=true`: `cpu` assembles and merges the full model on CPU; `npu` loads only this rank's base/branch shard and merges there. It cannot use `tp_cache`, and its NPU delta is non-bitwise; parity names are then experimental, not a bitwise claim |
| `parallel.merge_delta_device` | `"npu"` | with `merge_device="npu"`: `npu` computes `B @ A` on the NPU; `cpu` computes the FP32 delta on CPU and applies the BF16 update on the NPU (slower, bitwise in the measured `off` run) |
| `render.prompt_file`, `render.out` | required | the prompt cache and the mp4 |
| `render.num_frames` | `345` | aligned to 17n+5; 345 frames are 14.4 seconds at 24 fps |
| `render.width`, `render.height` | `1344`, `768` | target canvas; both axes must be multiples of 32, aspect ratio 1:4 through 4:1 |
| `render.fps` | `24` | export rate: 12, 24, 25, 30, 48, or 60; native generation remains 24 FPS |
| `render.num_steps` | `50` | model evaluations |
| `render.warmup_steps` | `0` | evaluations run and discarded first |
| `render.seed` | `42` | |
| `render.device` | `cuda:0` | `cuda[:index]` or `npu[:index]` |
| `render.offload_dit` | `false` | block-level CPU offload; supported on bf16 NPU/CUDA runs, including sharded two-NPU TP |
| `kernels.npu_resident_blocks` | `0` | NPU + `render.offload_dit` only: keep the first N DiT blocks in HBM and stream the rest; `0` keeps every block offloaded |
| `kernels.npu_pinned_offload` | `false` | NPU + `render.offload_dit` only: pin all CPU weights once and stream blocks over an NPU side stream; opt-in, requires a hybrid checkpoint, costs about 97 GiB of host pinned memory, and is mutually exclusive with `npu_resident_blocks` and TP |
| `render.video_shift`, `render.audio_shift` | `12.0`, `3.0` | scheduler shifts; a distilled adapter is valid only at the shifts it was trained at |
| `render.record` | `false` | also write `<out>.inference.json` |
| `render.save_latents` | `false` | also save the video and audio latents under `render.latents_root` |

`render.num_steps` is the number of model evaluations. The scheduler counts sigma grid points, one more than that, and the render passes the extra one itself.

Keyframe caches must be encoded for the selected target canvas: pass matching
`--width` / `--height` to `encode_keyframes.py`. The WebUI handles this automatically.
Ref2VA reference latents retain their own geometry. A non-native `render.fps` performs
nearest-frame export resampling and leaves audio on the native timeline; it does not
perform motion interpolation or change playback speed.

On Ascend NPU, use `render.device=npu:0` with `configs/inference/8nfe.yaml` or
`8nfe_tuned.yaml`. `auto` resolves the window softmax to `ref`; the explicit
`kernels.softmax_backend=npu_varlen` opt-in replaces it with grouped TND fusion-attention
calls (not bitwise). The NPU route runs the linear branch with eager PyTorch operators and
supports block CPU offload. The CUDA route keeps the Flex/FA4/Triton and FP8
implementations. Use `render.num_frames=39` or more when
testing the hybrid far branch; 22 is the smallest valid aligned frame count.

On a 64 GB NPU, `kernels.npu_resident_blocks=N` keeps the first N DiT blocks in HBM
and streams the rest, which cuts denoise time without changing the arithmetic. The value
is NPU-only, requires `render.offload_dit=true` and a hybrid checkpoint, and must leave
at least one block offloaded. Before the VAE onload, the resident blocks are moved back
to CPU and the HBM cache is released; this is automatic when the count is nonzero and also
applies to `infer_ulysses.py` (including ranks without a decoder). `0` is the old
all-offload path and remains the default.

The safe envelope measured at 1344x768 on a 64 GB NPU is duration-dependent:

| Frames | Residency | Measured result |
|---|---|---|
| up to 39 | up to 32 | 28/32 bitwise-identical to all-offload; 8 NFE denoise 1.63x/1.82x |
| 40–299 | 0 recommended | not measured; no validator gate, but nonzero values risk OOM |
| 300–345 | up to 16 measured | 12 recommended, 16 was the verified maximum (345-frame 8 NFE completed); larger values are allowed but may OOM |

At 345 frames, 28 blocks OOMed at 1344x768, and 20/22 blocks left only about 20 MiB of
HBM headroom. The table is a measured envelope, not a validator gate; larger frames or
block counts can OOM. Residency helps repeated
evaluations: at 39 frames the 1-NFE total wall is only about 1.1x faster because model
load and the pre-decode eviction are fixed costs, while the 8-NFE run is about 1.4–1.5x
faster.

`kernels.npu_pinned_offload=true` is a separate, default-off NPU path. It pins every
CPU parameter and buffer once at startup, then installs block-level offload with an NPU
side stream, `non_blocking=true`, `num_blocks_per_group=1`, and no `record_stream`. The
first forward keeps the top-level group synchronous; later forwards prefetch the next
block while the current one computes. Weights and arithmetic are unchanged. Measured at
1344x768 on a 64 GB NPU: 39-frame 1-NFE denoise went from 67.8 s to 17.6 s (3.85x) and
was bitwise identical (video and audio); 345-frame 1-NFE went from 214.0 s to 163.9 s
(1.31x). The cost is host memory: the 65.7 GiB transformer reserves about 97 GiB of pinned
host memory (the allocator rounds each tensor up), and pinning takes about 34 s. That
startup cost is why a single 1-NFE render may not improve end-to-end wall time; the gain
grows with the number of NFEs (39-frame 8-NFE denoise 547.3 s → 142.7 s). The loader
refuses to start unless host `MemAvailable` is at least 1.5x the model bytes plus 8 GiB.
The option requires `render.offload_dit=true` and a hybrid checkpoint, and cannot be
combined with `npu_resident_blocks`. Use the CLI, for example:

```bash
python src/inference/infer.py --config configs/inference/8nfe.yaml \
  checkpoint=ckpts/stage-dmd-step-250 render.prompt_file=prompts/example_0.pt \
  render.out=results/pinned.mp4 render.device=npu:0 render.offload_dit=true \
  kernels.npu_pinned_offload=true
```

## Assembly

Both entry points build the model through [src/inference/utils/assemble.py](../src/inference/utils/assemble.py), in this order: read the checkpoint's spec, load the base, apply the hybrid transform, load the linear branch, merge the checkpoint's LoRAs, merge the external LoRAs, select the inference kernels and the window-softmax backend, then convert to fp8. The record written by `render.record=true` states what actually ran: the checkpoint's identity, the resolved config, the resolved backend, the number of fp8 Linears, and the time of every evaluation.

## Inference kernels

`kernels.inference_kernels: true` swaps in forward-only kernels with the same arithmetic:

- **Attention:** QK-norm and RoPE fused in one pass over views of the packed q/k/v buffer, and the softmax gate fused with the output repack.
- **Linear branch:** one kernel for the frame statistics, a Triton 5-tap temporal convolution, a compiled RMSNorm and output gate, one triangular solve for the inverse, and `baddbmm` scan updates.
- **Block:** RMSNorm with the AdaLN affine, gate and residual as two compiled kernels, and SwiGLU as one.

The first run compiles them, which takes several minutes; later runs reuse the cache.

## Window softmax

Four implementations of the same windowed attention:

- `flex`: FlexAttention with a block mask. Training uses it.
- `decomposed`: the mask as a union of dense attentions, with no mask machinery. Global rows attend to everything in one dense call, and the frames that share a window go through one variable-length call.
- `ref`: the eager reference, for parity checks.
- `npu_varlen`: NPU-only, explicit opt-in. The dense leg uses plain SDPA over the full key/value, and each window group runs one `torch_npu.npu_fusion_attention` TND varlen call over its gathered kv, one group at a time to bound gather memory. The mask is exact, but the bf16 reduction order is not bitwise against `ref`.

Measured on 39f / 512x288 / 8NFE, two-NPU TP `off`: `npu_varlen` denoise was 44.77 s versus 45.77 s for `ref` (2.2% faster), with unchanged physical HBM (~38.5 GiB per rank). Video quality stayed in the `ref` error band, but audio latent error against the single-card reference was about 1.9x the `ref` baseline and decoded audio SNR fell 0.4 dB. It is therefore an experimental, explicit opt-in, not the default; `auto` on NPU remains `ref` for parity and audio quality.

`auto` is `decomposed` on every CUDA device. It is the faster kernel on B200 and matches `flex` on H200. The two differ by bf16 reduction order, and by their transient memory, which [Memory](#memory) measures. On NPU, `auto` stays `ref`; choose `npu_varlen` explicitly:

```bash
python src/inference/infer.py --config configs/inference/8nfe.yaml \
  checkpoint=ckpts/stage-dmd-step-250 kernels.softmax_backend=npu_varlen \
  render.device=npu:0 render.prompt_file=prompts/example_0.pt render.out=results/npu_varlen.mp4
```

Each CUDA implementation picks its kernel by the card:

| Card | `flex` | `decomposed` |
|---|---|---|
| Hopper, data-center Blackwell (sm90, sm100, sm110) | FlexAttention's Flash backend (FA4) | FA4's varlen kernel, cuDNN SDPA |
| Ampere, Ada, consumer Blackwell (sm8x, sm120) | FlexAttention's Triton kernel | PyTorch's `varlen_attn`, SDPA's own kernels |

Nothing degrades to dense attention. A missing package raises its own import error: `flash-attn-4` on the first row, PyTorch 2.13 on the second.

## fp8

`precision.fp8.enabled: true` replaces every Linear at least 4096 wide on both sides with [Fp8Linear](../src/models/ops/fp8_linear.py): the qkv and output projections, the feed-forward pair and the linear branch's readout, 363 Linears with `skip_end_blocks: 0`. Narrow Linears stay in bf16.

- The quantizer is a Triton kernel, one program per row. The attention quantizes its input once for the three projections, and the feed-forward fuses SwiGLU with the quantization of its output.
- Scales are per row on sm90 and per tensor on sm100 and above, which is where each card's fast GEMM is. The granularity follows the card and is not a knob.
- The swap is one way: the bf16 weight is released as each Linear is replaced, so the model is roughly half its weight afterwards.
- fp8 needs compute capability 9.0 or above.

fp8 changes the sample: one evaluation stays close to bf16 (cosine 0.998), and the trajectory settles into a different video of the same quality. A seed reproduces a render only within one precision.

## Eight GPUs

`infer_ulysses.py` shards the packed sequence by row across the ranks. Inside attention one all-to-all turns sequence-sharded, all-heads q/k/v into all-sequence, head-sharded q/k/v, and a second one restores the layout, so the window softmax and the linear branch both see the whole sequence. There is no KV cache, token dropping or approximate communication.

On Ascend NPU the default path is a sequence-gather fallback: each rank gathers the
packed residual, runs every block eagerly, and returns its local rows. It is
bitwise-identical to single-card sampling but does not reduce compute. The experimental
`parallel.npu_head_parallel=true` opts into the standard head-parallel Ulysses path
(HCCL all-to-all QKV, head-sharded softmax and eager far branch), requiring at least two
ranks and a head count divisible by the world size. In 2-rank measurements it removed
duplicated attention/linear compute but did not produce a useful speedup: each rank still
loads and streams the complete 65.7 GiB transformer, so weight movement dominates (39
frames at 512x288: 495 s versus 496 s at 0 resident blocks; 1.08-1.10x with 28/32 resident
blocks). The row-sharded GEMMs also changed bf16 reduction order, and the 8-NFE video did
not meet the near-lossless gate (latent relative L2 0.166, RGB PSNR 27.7 dB), so the
option remains off by default. A real multi-NPU speedup needs weight sharding per rank
rather than head parallelism over a complete copy of the weights.

The layout is branch-parallel: `parallel.softmax_ranks` ranks run the softmax heads and the rest run the linear branch. The shipped configs state the split and the kernel each card was tuned with:

| Config | Split | Window kernel |
|---|---|---|
| `_ulysses_h200.yaml` | 6 + 2 | `flex` |
| `_ulysses_b200.yaml` | 5 + 3 | `decomposed` |

Only rank 0 loads the decoders and writes the mp4. T2VA, keyframe and reference caches all run here: conditioning rows sit outside the video span, so they shard and attend as ordinary global rows.

Setup is dominated by eight ranks reading 66 GB each, so keep `ckpts/` on a local disk. The last log line splits the time into setup, denoise, and decode plus encode. Eight-GPU renders are not bit-reproducible from run to run; one-GPU renders are.

## Memory

The fp8 configs fit an 80 GB card, on one GPU and on eight. Peak GPU memory in GiB at 345 frames and 8 evaluations, on the T2VA example, measured on an H200 with the CUDA allocator capped at 78 GiB:

| Phase | One GPU | Eight GPUs, per rank |
|---|---:|---:|
| Load: the bf16 transformer | 61.7 | 61.7 |
| Assembly, through the fp8 conversion | 65.9 | 65.9 |
| Denoise | 71.4 | 55.9 to 60.4 |
| Decode | 67.8 | 68.5, rank 0 only |

The decoders wait on the CPU and move to the GPU when denoising is over, and the fp8 conversion releases each bf16 weight as it goes, so the model never sits on the card twice. Assembly then returns the freed memory to the driver, where NCCL allocates its buffers at the first collective. On one GPU the bf16 configs need more than 80 GB: 66 GiB of weights plus 26 GiB of transients.

The two encoders fit the same card: `encode_prompt.py` peaks at 63.2 GiB, and `encode_keyframes.py` with six references at 69.2.

Conditioning rows raise the denoise peak, and the window kernel decides by how much. `decomposed` gathers the global rows once per window group, and `flex` has no gather. One GPU, the same cap:

| Request | Window kernel | Denoise peak | Seconds per evaluation |
|---|---|---:|---:|
| T2VA | `decomposed` | 71.4 | 11.8 |
| T2VA | `flex` | 62.7 | 11.8 |
| FL2VA example | `decomposed` | 74.4 | 14.2 |
| Ref2VA-like example | `decomposed` | out of memory | 19.1 without the cap |
| Ref2VA-like example | `flex` | 63.8 | 17.8 |

`auto` stays `decomposed`. On an 80 GB card, render a reference cache with `kernels.softmax_backend=flex`:

```bash
python src/inference/infer.py --config configs/inference/8nfe_tuned_fp8.yaml \
  checkpoint=ckpts/stage-dmd-step-250 kernels.softmax_backend=flex \
  render.prompt_file=prompts/reference/example_ref2va.pt render.out=results/example_ref2va.mp4
```

## Prompt caches and conditioning

A prompt cache is one `.pt` holding the prompt text, the Qwen3-VL hidden state after layer 50 as `(L, 5120)` bf16, and a tag per row. A keyframe or reference cache adds the anchors and one VAE latent per image. Each token costs 10 KB, and an image costs about one token per 32×32 pixels, so a cache with images runs to tens of megabytes.

| Request | Encoder flags | Packed sequence |
|---|---|---|
| T2VA | `encode_prompt.py --prompt` | `[text, audio, video]` |
| I2VA, L2VA, FL2VA | `encode_keyframes.py --first` and/or `--last` | `[text, keyframes, audio, video]` |
| Ref2VA-like | `encode_keyframes.py --refs a.png b.png ...` | `[text, one block per reference, audio, video]` |

Keyframes go onto the video canvas: the first is stretched to it and the last is cover-cropped. Image references keep their own aspect ratio on a 768-pixel short edge (`--ref_size`), videos are normalized to 24 fps on their own canvas, and audio clips are resampled through the audio VAE. Visual references take their own rotary slots ahead of the video; video soundtracks and audio clips add clean audio rows at t = 1.0. Images and videos appear as `<Picture i>` / `<Video k>` vision blocks in the text; audio is labelled `<Audio j>` and never reaches the conditioner. On the hybrid model the conditioning rows are global tokens, attended densely in both directions by the window softmax and absent from the linear scan.

Conditioning rows lengthen the sequence. A T2VA render is about 104k audio and video rows plus the text. The Samoyed reference example adds 6,948 text and 5,664 reference rows, and runs at 19.1 seconds per evaluation on one H200 in fp8 and 3.4 on eight.

## Frames, canvas and seeds

The canvas is 1344×768. `render.num_frames` is aligned to 17n+5, and 345 frames become 102 latent frames.

One-GPU renders are bit-reproducible: the same config, seed and cache give the same mp4 bytes on the same GPU model. The initial noise differs between GPU models.

## Output

The mp4 is 24 fps with the audio muxed in. It is encoded to a sibling `.partial.mp4` and renamed into place, so a file under the final name is always complete.

## Weights

A path written as `ckpts/<name>` that does not exist locally is downloaded from the Hub into the Hugging Face cache and used from there, so every command in the README also runs without a local `ckpts/`.

`external_loras` merges community adapters for the dense MiniMax-H3, safetensors with peft names, after the checkpoint's own LoRA:

```bash
python src/inference/infer.py --config configs/inference/8nfe.yaml \
  checkpoint=null "external_loras=[{path: turbo.safetensors}]" \
  render.prompt_file=prompts/example_0.pt render.out=results/turbo.mp4
```

`alpha` defaults to the value in the file's metadata, and to the rank when the file has none.

## Timing a render

Set `render.warmup_steps: 2`, as the fp8 configs do. The first evaluation compiles, and the second re-specializes for the next timestep. The timing line reports every evaluation after that, and `render.record=true` keeps them in the JSON. The numbers in the README's [Results](../README.md#results) exclude model loading, warm-up, VAE decoding and mp4 encoding.

## Experimental: two-NPU weight tensor parallelism

`parallel.npu_tensor_parallel=true` explicitly enables the dedicated
[`infer_tp.py`](../src/inference/infer_tp.py) entrypoint. It defaults to **false**.
**Experimental: the default (`off`) real 8NFE video/audio comparison against
single-card inference FAILED the near-lossless gate.** Same-TP repeatability does not
establish single-card equivalence. The opt-in `row_gather`/`row_stream` parity modes
below are bitwise on the measured 39f canvas; see their separate limits. It supports
exactly two local NPU ranks and a BF16 hybrid checkpoint;
**4/8-card TP is unsupported**. It rejects head parallelism, teacher mode, FP8 and
the separate pinned-offload/resident-block overrides. `render.offload_dit=true`
enables synchronous CPU block offload of the TP shards. The original measured
resident small-canvas configuration is:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/inference/infer_tp.py checkpoint=ckpts/stage-dmd-step-250 \
  base_source=ckpts/h3-base vae_source=ckpts/h3-base \
  parallel.npu_tensor_parallel=true kernels.inference_kernels=false \
  render.device=npu:0 render.prompt_file=prompts/example_0.pt \
  render.out=results/tp-39f-512x288-8nfe.mp4 \
  render.num_frames=39 render.width=512 render.height=288 \
  render.num_steps=8 render.seed=42 render.warmup_steps=0 \
  render.offload_dit=false render.record=true render.save_latents=true
```

By default (`parallel.merge_device="cpu"`) assembly and LoRA merging happen on CPU,
serialized by rank. Each sliced parameter is cloned before its full source is
released; no sharded checkpoints are written. Only this entrypoint temporarily uses
at most eight CPU intra-op threads, bounded by CPU affinity and visible cgroup
quotas, during CPU assembly/merge/shard/cleanup. It restores the original torchrun
thread count in `finally`, before NPU onload, denoising or decoding, including on
assembly failure. It changes no thread environment variables and adds no
configuration switch. The serialized guard remains in place: rank0 completes onload
before rank1 constructs its full CPU model.

### CPU block offload for TP

`render.offload_dit=true` installs the existing synchronous Diffusers block-level
offload hooks **after** TP replacement and parity preparation. Each rank loads the
current block's shard, runs exactly the same TP operations, then returns that block
to CPU. The small root groups stay on device during the root forward. All module
parameters and buffers are back on CPU between forwards. `row_gather`'s complete
Row weights are nonpersistent registered buffers; they are included in this path.
Runtime parity scratch remains separately managed and is released before decoding.

This follows the on-demand weight-residency principle found in the local
[`comfy-aimdo` NPU fork](https://github.com/vtloader/comfy-aimdo/tree/npu-support),
particularly `comfy_aimdo/model_vbar.py` and `examples/example.py`. It does **not**
install AIMDO's allocator or virtual-address/page cache. Two TP-local strategies are
available under `parallel.tp_offload_strategy`:

- `retained` (default): immutable CPU backing is captured after every merge. Onload
  binds device copies; after the compute stream completes, eviction restores the CPU
  references without copying unchanged weights back. This removes the old synchronous
  D2H traffic.
- `prefetch`: retained backing plus one-block lookahead. CPU backing is pinned once,
  the next block's H2D runs on a private copy stream, and at most two device block
  leases are retained. HCCL and model operations stay on the existing compute stream
  and in the existing collective order. This lowers exposed transfer stalls but costs
  roughly 36 GB pinned host memory per rank. Both ranks pin their own backing
  concurrently, so the pin step is about 19 s instead of 36 s; total prefetch model
  setup measured **37.6 s** at 39f/1NFE (13.0 s cache read + 19.3 s pin).

The old copy-back behavior is available as `synchronous` for controlled comparisons.
`prefetch` is the WebUI fast preset; the CLI/config default remains `retained`. Both strategies preserve the existing GEMM shapes.

For CPU merge/cache with `row_stream` or `off`, the shard stays on CPU until used.
NPU merge still temporarily onloads the shard to merge LoRAs, then moves it back to
CPU before rendering. `row_gather` prepares its complete weights on NPU with both
ranks active before offload hooks are installed. Merge arithmetic/quality is
independent of offload: use CPU merge/cache or CPU delta when CPU-merge parity matters.

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/inference/infer_tp.py --config configs/inference/8nfe.yaml \
  checkpoint=ckpts/stage-dmd-step-250 parallel.npu_tensor_parallel=true \
  parallel.npu_tp_parity=row_stream parallel.merge_device=npu \
  render.device=npu:0 render.offload_dit=true \
  render.prompt_file=prompts/example_0.pt render.out=results/tp-offload.mp4 \
  render.num_frames=243 render.width=1920 render.height=1088 render.num_steps=1
```

The inference JSON records `parallel.cpu_offload`, `parallel.offload_strategy`,
per-rank module storage placement before/after denoise, parity scratch bytes and
allocator peaks by phase. Storage placement is logical tensor storage, distinct
from allocator-reserved or driver-reported HBM. TP emits `[vdn-progress]` loading,
denoise-step and decode events for CLI/WebUI observability.

Setting `VDN_TP_WORKER_QUEUE=<dir>` turns `infer_tp.py` into a resident worker:
rank zero polls `<dir>/request.json`, both ranks render the request, and rank zero
writes `<dir>/result.json` while keeping the transformer, HCCL group and CPU/pinned
backing alive for the next request. `<dir>/ready` marks a completed model load and
`<dir>/stop` shuts the worker down. The WebUI uses this mode for TP tasks, so only the
first generation pays the model setup (about 38 s with the cache warm); later
generations with the same fixed settings reuse the worker. Requests may still change prompt, steps, frames, size, FPS
and seed. Fixed settings are checkpoint, device, sampling config, `offload_dit`, TP
parity/merge/cache and offload strategy; changing any of them restarts the worker.
The WebUI also caches prompt encodings by mode/prompt/images/model, so a repeated
prompt skips the text encoder subprocess.

CPU/Gloo tests and the two-NPU `tests/npu_tp_offload_smoke.py` compare resident TP
against offloaded TP in the same parity mode: all three modes, FP32/BF16 on NPU,
four repeated forwards, parameter/buffer placement and exact video/audio equality.

Controlled 39f/512x288/4NFE A/B, CPU merge/cache, `row_stream`, NPU delta:

| Strategy | Denoise | Setup | Decode | E2E |
|---|---:|---:|---:|---:|
| `retained` | **72.27 s** | 17.87 s | 12.89 s | **123.27 s** |
| `prefetch` | **25.36 s** | 64.14 s (44.42 s pin) | 52.55 s | 149.53 s |

Both produced byte-identical video/audio latents. Prefetch cuts denoise time by
**2.85x**, but its one-time pin and a slower decode make E2E worse for this short
4-step request; it is a utilization/denoise optimization, not automatically an E2E
win. Host pinning used 36.21 GB logical backing per rank and the allocator reported
53.64 GB reserved per rank. The pin is now concurrent across the two ranks, which
cuts its measured share from 36.6 s to 19.3 s and total setup from 65.0 s to 37.6 s
on the 39f/1NFE probe. Peak device leases were 2, with 1.38 GB of device block
copies. Evidence: `/tmp/opencode/tp_offload_duty_prefetch/`.

The same controlled A/B at **8NFE** shows the intended multi-step gain:

| Strategy | Denoise | Setup | Decode | E2E | AICore proxy NPU0/NPU1 |
|---|---:|---:|---:|---:|---:|
| `retained` | 140.75 s | 18.59 s | 11.53 s | 342.80 s | 13.10% / 8.87% |
| `prefetch` | **50.78 s** | 60.92 s | 47.91 s | 167.15 s | 36.93% / 31.53% |
| `prefetch` + pre-decode release | 51.44 s | 70.03 s | **27.20 s** | **148.95 s** | — |

Latents were byte-identical. Prefetch is therefore recommended for longer
multi-step renders and is the WebUI default; the config default remains retained
because a one-step request can be faster without pinning. AICore values are sampled
proxies, not exact kernel duty.
Evidence: `/tmp/opencode/tp_offload_duty_8nfe/summary.json` and
`/tmp/opencode/tp_prefetch_release_39_8nfe.mp4.inference.json`.

After denoise, the TP path now drops the transformer and its CPU/pinned backing
before VAE decode; decode only needs the two VAEs. This reduced the 39f/8NFE prefetch
decode from 47.91 s to 27.20 s and E2E from 167.15 s to 148.95 s without changing
latents. It is safe because `render_record` needs artifact/config metadata, not the
transformer, and no forward follows. The remaining decode overhead is the host
pinned-allocator cache, which cannot be explicitly released on this TorchNPU build.

### NPU triangular inverse

`torch.linalg.solve_triangular` has no NPU kernel. In a one-step profile it fell back
to AiCPU and consumed **3.86 s of 7.86 s** at 39f/512x288, leaving the NPU idle between
DiT blocks. `_triangular_inverse` now inverts each 16x16 diagonal block by forward
substitution, then merges blocks with the Schur complement `-D^-1 C A^-1` on NPU GEMMs.
It computes the same exact inverse with a different rounding association; the maximum
difference from CPU `solve_triangular` on the first six real 128x128 systems was
**3.6e-7**. CUDA and CPU keep the cuSOLVER/CPU trsm path.

Measured with CPU merge/cache, `row_stream`, prefetch, and pre-decode release:

| Geometry | Denoise before | Denoise after | AICore NPU0/NPU1 before | AICore after |
|---|---:|---:|---:|---:|
| 39f / 512x288 / 8NFE | 51.44 s | **21.64 s** | 36.93% / 31.53% | **86.50% / 85.14%** |
| 243f / 1920x1088 / 1NFE | 134.33 s | **110.61 s** | — | **93.19% / 93.68%** |

AICore values are sampled `npu-smi` proxies over the device-synchronised denoise
window, not kernel duty. The 1080p step still peaks at **39.39 GB allocated /
63.90 GB reserved** per rank. At 39f/8NFE the new output is visually matched but not
bitwise against the CPU-trSM baseline: decoded video PSNR **22.72 dB**, SSIM
**0.9184**, audio SNR **9.97 dB**. The small inverse-rounding difference is amplified
by the 8-step distilled sampler; per-layer inverses agree to 3.6e-7. Evidence:
`/tmp/opencode/tri_block_39_8nfe.mp4`, `/tmp/opencode/baseline_39_8nfe.mp4`,
`/tmp/opencode/measure_aicore_39_8nfe.json`,
`/tmp/opencode/measure_aicore_1080_1nfe.json`, and
`/tmp/opencode/tri_block_1080_1nfe.mp4.inference.json`.



Measured full-model T2VA at **1920x1088 / 243f / 1NFE / seed42**, bundled
`prompts/example_0.pt`, `row_stream`, CPU merge/cache and NPU delta, `prefetch`:
setup **65.11 s**, denoise **110.61 s**, decode/encode **110.14 s**, E2E **286.59 s**.
Before the blocked triangular inverse the same prefetch geometry measured **134.33 s**
denoise; retained measured **142.24 s** before that change. Per-rank retained denoise peaks
were **38.69 GB allocated / 62.49 GB reserved** (decimal GB); prefetch added about
0.69 GB allocated / 1.41 GB reserved for the second lease.
Both ranks' 36.21 GB of module weights/buffers were on CPU before and after denoise;
parity scratch was 0.91 GB. Sampled physical HBM reached about 63,440/63,184 MiB,
so allocator-reserved/physical occupancy remains high despite much lower live weight
residency. Sampled compute intervals reached 98–100% AICore, alternating with
transfer/wait intervals. This is a memory/completion result, not full 8NFE quality
validation. Evidence: `/tmp/opencode/tp_offload_1080_243_1nfe.mp4.inference.json`.

The WebUI generates a missing or checkpoint-mismatched `tp_cache` automatically before
the worker starts; an empty path still means "assemble on every launch".

A synthetic FL2VA shape-pressure test also completed **243f/1920x1088, two steps**:
4398 prompt rows made by repeating the bundled sample embeddings, plus two zero
first/last keyframe latents of shape `(1,24,1,68,120)`. This tests request size and
repeated offload, not visual quality. Setup **66.04 s**, steps **178.05/175.73 s**,
decode/encode **111.51 s**, E2E **532.05 s**. Rank-zero peaks were **40.58 GB allocated /
64.13 GB reserved**; sampled physical HBM reached about **65,008/64,749 MiB**. This
still has very little driver-visible headroom and is not an 80%-HBM acceptance result.
Evidence: `/tmp/opencode/tp_offload_fl2va_shape_1080_243_2nfe.mp4.inference.json`.

Full-model numerical regression at **39f/512x288/8NFE**, CPU-merged cache,
`row_stream` and block offload: video/audio latents were **bitwise identical** to
the existing parity reference (`max_abs=0`), and the MP4 SHA-256 also matched:
`04d0a064204f385b20cb4fe6ecd209bca6cb27d8c449ed84e7872cdd97d209f5`.
Evidence: `/tmp/opencode/tp_offload_cache_parity.json`. The first step took 212 s,
then subsequent steps about 35 s. CPU-cache offload defers mmap page access to the
first forward, and the regression suite ran concurrently with part of this probe;
these timings are observational, not a controlled throughput comparison. The
large-geometry runs above did not overlap that regression suite.

### Direct base-shard load and NPU merge (`parallel.merge_device="npu"`)

`parallel.merge_device="npu"` removes the full CPU assembly from the launch path.
Both ranks start together (there is no rank-serialized loop): each builds the hybrid
transformer skeleton on `meta`, installs the same TP layout, and reads only its final
parameter shard directly from the base safetensors files and the checkpoint's
`linear_branch/model.safetensors`. `ColumnLinear` reads its output rows, `RowLinear`
reads its input columns, directly sharded branch tensors (`alpha.dt_bias`,
`alpha.A_log`, the short-conv weights) read one first-dimension interval, and
replicated parameters read the complete tensor. The resulting 33.7 GiB state dict is
assigned onto the skeleton and moved to the NPU. The checkpoint's 571 LoRA pairs and
any external adapters are then merged on the device with
`merge_lora_state_npu`; `merge_lora_state` never runs on this path, and no 67 GiB
merged cache is written.

The option requires `npu_tensor_parallel=true` and an empty `parallel.tp_cache`.
The device-side BF16 merge is not bitwise identical to the CPU merge, so a parity
label is not a bitwise claim on this path; use CPU merge or the precomputed cache for
bitwise parity. Frame count and canvas size are not gated here. The inference JSON records `parallel.merge_device` and, per rank, the
`base_read_seconds`, `assign_seconds`, `onload_seconds`, `npu_merge_seconds` and
`merge_delta_device` spans.

Measured on 39f / 512x288 / 8NFE with the base files in page cache, using the default
`merge_delta_device="npu"`: base-slice read **12.9–18.4 s** per rank, H2D
**15.4–16.0 s**, NPU merge **1.1–1.5 s**, model setup **54.9 s**, E2E **128.6 s**,
physical HBM **40.3/40.7 GiB**. The first cold read can be slower. This path is not
bitwise against the CPU merge: video latent relative L2 was **0.150** versus the
CPU-merge `off` run, RGB global PSNR **23.74 dB**, mean SSIM **0.898**, and decoded
audio SNR **9.99 dB** (CPU `off` against the same single-card reference: 25.80 dB /
0.929 / 10.07 dB).

`merge_delta_device="cpu"` keeps the FP32 `B @ A` delta exactly as the CPU merge and
applies the BF16 update on the NPU. Measured setup was **206.1 s**, E2E **279.4 s**,
and video/audio latents were `torch.equal=True` against the CPU-merge `off` run. This
is the quality-safe no-cache option; the default NPU-delta path is the fast option
and is not a parity path.

#### Precomputed merged shard cache (`parallel.tp_cache`)

The CPU assembly and every LoRA merge can be paid once instead of at every launch.
Generate the cache for one exact checkpoint/base/external-LoRA combination:

```bash
.venv/bin/python -m src.inference.prepare_tp_cache --out /path/to/tp-cache \
  --config configs/inference/8nfe.yaml checkpoint=ckpts/stage-dmd-step-250 \
  base_source=ckpts/h3-base vae_source=ckpts/h3-base
```

The generator runs entirely on CPU, rank 0 then rank 1, and forces the
cache-generation-safe values (`render.offload_dit=false`,
`kernels.npu_resident_blocks=0`, `kernels.npu_pinned_offload=false`,
`precision.fp8.enabled=false`, `kernels.softmax_backend="ref"`,
`parallel.npu_tp_parity="off"`, `parallel.merge_device="cpu"`). It writes:

```
cache_dir/
  manifest.json
  rank0.safetensors
  rank1.safetensors
```

`manifest.json` records format/kind/world size/dtype, the absolute checkpoint path
and its `checkpoint_head_sha256`, the full `model_spec`, checkpoint artifact metadata,
resolved base and VAE sources, the external-LoRA records, merged pair count, per-rank
file names and sizes, each rank's `install_tensor_parallel` scope/layout, and the
generator version. A non-empty `--out` is refused unless `--force`; `--force`
overwrites only the three known cache files.

Point `infer_tp.py` at it with `parallel.tp_cache=<cache_dir>`. Every rank then builds
the transformer skeleton on `meta`, installs TP without loading or merging base
weights, loads only its own rank shard, and moves it to its NPU. `merge_lora_state`
never runs. The manifest is validated at runtime (checkpoint hash, base source, LoRA
set and rank files), and `parallel.tp_cache` plus per-rank cache-load seconds are
recorded in the inference JSON.

The cache is only for the measured two-NPU BF16 TP path. Each rank file is roughly
half of the 65.7 GiB merged transformer plus the replicated layers (about 33 GiB), so
the cache directory is about 66 GiB; `rank_file_bytes` in the manifest gives the exact
sizes. It stores no decoder or text-encoder weights.

Measured on 39f / 512x288 / 8NFE with two 64 GiB NPUs: cache generation took
**363.2 s** and produced **67.45 GiB** (two 33.72 GiB rank files). The first cached
launch (cold file read, including rank-0 decoder load) had **134.0 s** setup and
**211.9 s** E2E; a second launch with the files in page cache had **34.2 s** setup and
**106.7 s** E2E, versus **538.1 s** for the original `off` launch. The cached `off`
video/audio latents were `torch.equal=True` and the MP4 SHA-256 was identical to the
original `off` run. The same cache also reproduced the bitwise `row_gather` result:
**34.5 s** setup / **119.4 s** E2E, with identical latents and MP4 hash. Startup is
now dominated by the rank-shard read and H2D transfer, not LoRA merging.

Every rank keeps its weight shard resident throughout denoising and holds the full
sequence residual. QKV and head-dependent gates, alpha and depthwise convolution
channels are head-sharded. FFN columns use matching intervals from both SwiGLU
halves. Attention/linear-branch/FFN output projections use input-channel shards,
FP32 sum reductions, and one bias addition after reduction. AdaLN output rows are
sharded and their small timestep tables gathered in original feature order before
the existing modality reshape/chunk. Norms, input/output heads, timestep layers,
low-rank gate/alpha down projections and the token refiner remain replicated.

The launcher is bounded to one node with matching global/local ranks 0/1 and
`LOCAL_WORLD_SIZE=2`; `render.device=npu:0` is the base device and each worker selects
its local NPU. The process group has a 45-minute timeout (allowing cold serialized
host loading) and is destroyed in `finally`. Installation requires `model.eval()`;
every root forward also rejects training mode or enabled gradients. Direct TP
projection calls reject enabled gradients. Use `torch.no_grad()` or inference mode.
Transformer inputs must be keyword tensors with matching shapes/dtypes on both
ranks. Only `attention_kwargs=None` or `{}` is supported; runtime LoRA scaling or
nested attention kwargs are rejected before broadcasting.

Rank zero broadcasts transformer request tensors each NFE, including initial noisy
rows, conditioning, timesteps and layout. All ranks release their transformer shards
from HBM before rank-zero-only decoding. The inference JSON records
`parallel.kind="npu_tensor_parallel"`, per-rank parameter/storage bytes, allocator
peaks, collective event timings, and setup/denoise/release/decode timings. The entrypoint
writes this evidence JSON even if `render.record` is false.

Records include exact sharded/replicated parameter-name scopes, global/local
head ranges per block, real on-device storage and an all-parameters-resident check.
`parallel.by_rank[].cpu_setup` records original/assembly/restored thread counts,
checkpoint/model loading, hybrid transform, branch loading and LoRA merge spans,
total build, shard, cleanup, synchronized NPU onload, each rank's active setup, and
setup including rank wait. Nested build spans must not be summed with total build.
Loading is mmap/lazy: these load spans are not complete storage-read measurements;
merge/shard/onload spans can include page faults.
They also include per-phase setup/denoise/release/decode allocator peaks and their
end-to-end maximum through the decode barrier (before final JSON metadata gathering).
Communication event time covers row-output reductions, Row all-gathers and AdaLN
gathers, including stream waits; it excludes request broadcasts, barriers and
metadata object gathers. In default (`off`) TP, BF16 local `F.linear` partials round
**before** casting to FP32 for SUM; this is not an FP32 local GEMM implementation.

This changes BF16 GEMM shapes and partial-sum rounding. Mathematical equivalence does
not imply video-lossless output; measure drift against a matching single-card run.

### Opt-in single-card parity: `parallel.npu_tp_parity`

Default is `"off"` (the split-K/partial arithmetic above). Two opt-in modes restore
the single-card BF16 GEMM shapes. Both are valid only with
`parallel.npu_tensor_parallel=true` and BF16 weights:

- `row_gather`: Column projections zero-pad the local rows to the complete output
  width in a bounded, shape-keyed scratch buffer and run a complete-N GEMM; AdaLN runs
  the same complete-N GEMM for its local rows and gathers only the tiny ordered table.
  Row projections AllGather the local activation, use a complete-K weight gathered once
  at onload, and run one complete-K GEMM with one bias. The redundant local Row column
  shard is released after the gather, so resident growth is half a complete Row weight
  per rank (measured +7.71GB, not +15.4GB).
- `row_stream`: the same Column padN arithmetic, but no complete Row weight is
  resident; each forward AllGathers the local Row weight and activation, runs the
  complete-K GEMM, and drops the temporary. It is the low-HBM option.

YAML/dotlist note: quote `"off"`; unquoted `off` is parsed as boolean false.
Example:

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/inference/infer_tp.py checkpoint=ckpts/stage-dmd-step-250 \
  base_source=ckpts/h3-base vae_source=ckpts/h3-base \
  parallel.npu_tensor_parallel=true parallel.npu_tp_parity=row_gather \
  kernels.inference_kernels=false render.device=npu:0 \
  render.prompt_file=prompts/example_0.pt render.out=results/tp-parity.mp4 \
  render.num_frames=39 render.width=512 render.height=288 \
  render.num_steps=8 render.seed=42 render.warmup_steps=0 \
  render.offload_dit=false render.record=true render.save_latents=true
```

#### Measured 39f / 512x288 / 8NFE parity results

Same seed/model/geometry as the single-card reference, two 64GiB NPUs, one run per
mode. The reference is `/tmp/opencode/dualnpu/single-39f-8nfe.mp4`; single-card
denoise was 496.150s. Physical HBM is the sampled `npu-smi` peak; percent uses 64GiB.

| Mode | Setup s | Denoise s | Median step s | Decode s | E2E s | Alloc/reserved peak per rank | Physical HBM peak (chip0/1) | vs single | vs TP off |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `off` | 465.21 | 45.771 | 5.658 | 10.35 | 538.06 | 36.75 / 37.32GB | 39,416/39,154MiB (60.1%) | 10.84x | 1.00x |
| `row_gather` | 465.66 | 49.482 | 6.129 | 10.40 | 545.61 | 45.66 / 50.09GB | 51,597/51,334MiB (78.7%) | 10.03x | 0.925x |
| `row_stream` | 462.76 | 50.539 | 6.240 | 10.34 | 541.23 | 37.89 / 38.89GB | 40,911/40,650MiB (62.4%) | 9.82x | 0.906x |

Setup/E2E are dominated by rank-serialized CPU assembly and vary by tens of seconds;
denoise is the controlled speed metric. The `off` repeat was bitwise identical to the
earlier `off` run; its failing quality against single-card is unchanged.

Quality against the single-card reference is **bitwise** for both parity modes:
video and audio latents `torch.equal=True`, max absolute difference 0, relative L2 0,
RGB PSNR infinite, SSIM 1.0, audio PCM SNR infinite, and the MP4 SHA-256 is identical
(`04d0a064...09f5`). Both pass the strict gate; the default `off` mode remains the
failing 0.110730 video relL2 / 25.80dB PSNR / 0.891 SSIM path.

**Recommendation: `row_gather` for the validated 39f/512x288 envelope.** It is the
fastest qualifying mode in the controlled metric (10.03x vs single-card denoise; 8.1%
slower than TP `off`) and stays under the 80% physical target, though with only ~1.3
percentage points of headroom. Use `row_stream` instead when HBM margin matters: its
denoise is 2.1% slower than `row_gather`, it uses 10.7GB less physical HBM, and its
E2E was 4.4s faster in this pair (setup-dominated, not a controlled comparison).
The following historical production-resolution measurements used **resident** TP,
before the CPU block-offload path was added:

| 345f / 1344x768 / 1NFE | Result |
|---|---|
| `row_gather` | **OOM** during denoise (58.00GiB allocated, physical peak 65,510MiB) |
| `row_stream` | **OOM** during Column padN FFN up (8.39GiB request, physical peak 65,516MiB) |
| `off` | completes: setup 474.14s, denoise 87.223s, decode 79.17s, E2E 657.40s; allocator reserved 60.07GB (87.4%); physical peak 61,132/60,870MiB (93.3%/92.9% raw, ~88.5% after the ~3.0GiB idle baseline) |

Therefore resident parity at 345f is **not safe**: both parity modes exceed the hard gate by
OOM, and even `off` is above the 80% target and at/over the raw 90% physical gate.
The validator no longer rejects large-frame parity, but the measured 345f parity
result is still OOM. Treat the 39f/512x288 numbers above as the measured envelope,
not a safety gate; oversized runs fail at runtime.
Session evidence: `/tmp/opencode/tp_parity/run_{off,row_gather,row_stream}/`,
`/tmp/opencode/tp_parity/compare_*.json`, and the `prod_*` 345f OOM/off runs.

### Measured 39f / 512×288 / 8NFE results

Seed42, `example_0.pt`, `stage-dmd-step-250`, BF16 eager, two 64GiB NPUs,
warmup=0, resident shards and rank0-only decoding. These replace the earlier 1NFE
smoke as the performance/quality reference:

| Stage (seconds) | Original TP, CPU threads=1 | CPU-assembly threads=8 |
|---|---:|---:|
| Cold model setup, both ranks | 922.615 | 492.585 |
| First synchronized 8NFE denoise | 45.816 | 45.774 |
| Release before decode | 16.775 | 18.886 |
| Decode/encode | 10.446 | 11.255 |
| Native-main cold E2E | 995.669 (repeat/diagnostics removed estimate) | 568.523 (one actual request) |
| Supervised process wall | 1028.701 (single-request estimate) | 591.880 (one actual request) |

Cold setup improved 1.87× (46.61% less time); **45.8 seconds is denoising, not
E2E**. Historical single-card group-offload denoising was 496.150s (about 10.84×
the new TP denoise time); this compares resident TP with CPU-offloaded execution,
not pure parallel-algorithm acceleration. Old raw process wall was 1074.202s and
included an immediate second 8NFE sample; the table's old E2E estimates remove
that repeat/diagnostics. These are observational starts, not a controlled cold-cache
storage benchmark; no OS page-cache eviction was performed for the CPU-thread run.

CPU LoRA merge: rank0 430.204→199.030s, rank1 388.344→210.194s. CPU clone/shard:
13.339→5.022s and 13.450→5.245s. New synchronized onload: 19.539/19.960s.
The CPU probe used four actual checkpoint pairs, 1/2/4/8/16 threads and both BF16
and FP32 destination weights: all merged weights were `torch.equal` to one-thread
`merge_lora_state`. BF16 four-pair mean-time sum was 3.995/2.974/2.409/2.150/2.068s.
Most of this improvement was in scale/cast/add; measured `B @ A` time was essentially
unchanged. The probe does not establish disk I/O dominance.

The new run's saved video/audio latents are `torch.equal` to the original TP 8NFE
latents (max absolute difference=0); the MP4 files are also byte-identical. Thus the
original single-card quality failure still applies: video latent relative L2
**0.110730** (gate ≤0.01), audio latent relative L2 **0.019212** (gate ≤0.01), RGB
global PSNR **25.797dB**, frame-mean PSNR **29.606dB** (gate ≥40dB), windowed SSIM
**0.891274** (gate ≥0.99), and decoded audio PCM SNR **10.069dB** (gate ≥40dB).
Keep the default `off`; the opt-in `row_gather`/`row_stream` modes above are the only
bitwise configurations, and their production-resolution limits are documented there.

Actual unique resident storage is **36,210,867,744 bytes (33.72GiB) per rank**,
including replicas, versus full **70,559,750,144 bytes**. New per-rank denoise
allocator peaks are 36,749,148,672 allocated / 37,318,819,840 reserved bytes;
sampled physical HBM peaks are 39,417/39,153MiB. Sampled host RSS sum peak is
105.08GiB, PSS sum 100.00GiB, and minimum `MemAvailable` 121.72GiB. RSS sums can
double-count shared pages; these sampled peaks are not continuous maxima.

Session evidence (local artifacts):
`/tmp/opencode/tp_cpu_startup_20261020_01/acceptance.json`, `cpu_probe.json`, and
`tp_cpu8_39f_512x288_8nfe.mp4.inference.json`. Original independent quality evidence:
`/tmp/opencode/tp_stage2/accept_20261020_01/acceptance.json`.

Tiny real-model NPU A/B checks cover FP32/BF16, full-cover attention and window
attention with short convolution and text state; run them with:

```bash
.venv/bin/python -m pytest tests/test_tensor_parallel.py
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/npu_tensor_parallel_smoke.py
```

The smoke writes its metrics to `/tmp/opencode/tp_stage1/npu_smoke_hardened.json` (create that
directory first when running outside the implementation session). Keep the default
disabled: the independent real multi-step near-lossless gate has failed; broader
geometry support has not been established.
It deliberately perturbs rank1 inputs and checks in-place caller noise views,
all request broadcasts, replicated block residuals, outputs and real video/audio
scheduler updates for cross-rank bitwise agreement, in addition to unsharded A/B.
