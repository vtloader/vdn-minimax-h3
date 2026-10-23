# VDN-H3 browser WebUI

A Python standard-library HTTP server, HTML, CSS, and JavaScript front end for the
repository's single-device inference entry points. No additional UI dependencies are
needed. Real generation uses the existing model environment and weights described in
the [repository setup instructions](../README.md#set-up-the-environment).

## One-click launch

From the repository root:

```bash
bash scripts/launch_webui.sh
```

Then open **<http://localhost:3000>**. The launcher activates the repository's `.venv`
when present and uses that Python for the server, both encoders, and the renderer.
Without `.venv`, it uses `python` from the current environment. An explicit `PYTHON_BIN`
can override the interpreter.

The script also works from another working directory when invoked by its full path:

```bash
bash /path/to/vdn-minimax-h3/scripts/launch_webui.sh
```

Set `HOST` and `PORT` to change the default `0.0.0.0:3000` bind address:

```bash
HOST=127.0.0.1 PORT=8080 bash scripts/launch_webui.sh
```

## Generation

| Mode | Required uploads | Encoder |
|---|---|---|
| `t2va` | None | `encode_prompt.py` |
| `i2va` | One first image | `encode_keyframes.py --first` |
| `fl2va` | One first and one last image | `encode_keyframes.py --first --last` |
| `ref2va` | up to 9 images, 3 videos, 3 audio clips (audio needs an image/video) | `encode_keyframes.py --refs/--video_refs/--audio_refs` |

Reference upload order is kept, matching `<Picture 1>`, `<Picture 2>`, `<Video 1>`,
`<Audio 1>`, and so on in the prompt. Images, videos and audio are packed in that
modality order. Selecting more references appends instead of replacing; every preview
has **替换** and **删除** buttons, and **清除图片/视频/音频** clears a whole group.
First/last images remain single-value. Ref2VA-like generation uses the existing FL2VA
transformer and checkpoints.
For image modes, follow the [prompt-writing instructions](../README.md#supporting-fl2va-i2va-and-l2va).
Prompts are passed verbatim and are limited to 20,000 characters.

The page keeps the current prompt, all form settings, and the selected reference images
in the browser (sessionStorage for fields, IndexedDB for images), so going to the task
log and coming back restores the form instead of starting over. Uploads support PNG,
JPEG, WebP, and BMP with basic file signature validation. The
encoder performs full media decoding. Images are limited to 20 MiB each, videos to
256 MiB, audio to 64 MiB, and the entire multipart request to 512 MiB. Video/audio
references need PyAV; audio resampling also needs `torchaudio` (install the build that
matches your PyTorch). Ordinary names, spaces, and Unicode names are preserved;
path components and unsafe characters are sanitized. Name collisions get numeric
suffixes such as `image_2.png`.

Defaults are `npu:0`, **1920 × 1088**, **10.125 seconds**, native/export **24 FPS**,
and the **8-step accelerated** checkpoint. Choose the sampling scheme independently of
the T2VA/I2VA/FL2VA/Ref2VA conditioning mode:

| Sampling scheme | Checkpoint | Recommended steps |
|---|---|---:|
| Accelerated | `ckpts/stage-dmd-step-250` | 8 |
| Standard | `ckpts/stage-b-step-2000` | 50 |
| Custom | Editable checkpoint path | User selected |

Preset changes update the checkpoint and recommended steps. Steps remain editable from
1–100. A preset cannot be paired with a conflicting checkpoint; select Custom for
other weights. Relative checkpoint paths are resolved from the repository root.

Resolution presets include 1344×768, 896×512, 1280×736, 768×1344, and 768×768. Custom
width and height must be multiples of 32 between 256 and 2048, with aspect ratio 1:4
through 4:1. Keyframes are center cover-cropped to the selected target canvas before
both vision encoding and VAE encoding; reference images retain their own geometry.
Uploaded images are previewed locally with their names, dimensions, and reference
numbers. Replacing/clearing files releases the preview URLs.

Duration accepts 1–15 seconds and is rounded to the nearest valid `17*n+5` native frame
count (at least 22). The UI displays the actual aligned duration. For example, 10 seconds
becomes 243 native frames / 10.125 seconds; 14.4 seconds becomes 345 / 14.375 seconds.
The model's native video and audio timeline is fixed at 24 FPS. Export FPS options are
12, 24, 25, 30, 48, and 60; non-native export rates drop/duplicate decoded frames, without
optical-flow interpolation or audio speed changes. Output duration can differ by at
most half an output frame due to rounding. Frame counts supplied through the legacy
API remain supported when `duration` is absent.

NPU requests use bf16 `8nfe.yaml` / `50nfe.yaml` and default to block-level CPU
offload. CUDA uses the same safe configuration unless **CUDA FP8** is explicitly
selected, which switches to the corresponding `*_tuned_fp8.yaml`. FP8 is rejected for NPU,
and cannot be combined with CPU offload. Offload applies to both the rendering transformer
and the Qwen3-VL encoder (leaf-level CPU offload during text/vision encoding).

On NPU, **NPU resident blocks** is an optional advanced setting. `0` keeps the old
all-offload path. Keeping some DiT blocks in HBM speeds up repeated model evaluations
without changing the arithmetic; resident blocks are moved back to CPU automatically
before VAE decoding. The measured 1344×768 envelope is 28–32 blocks at 39 frames and
12–16 at 300–345 frames, but the WebUI no longer gates frame count or block count
beyond the input range 0–49; larger values can OOM. It mainly helps multi-step runs:
at 39 frames, 8-NFE total wall improves about 1.4–1.5×, while a single evaluation is
dominated by model load and decoding.

**Experimental dual-NPU tensor parallel (TP)** is enabled by default in the WebUI,
using the fastest 1080p test preset: **1920x1088 / 243 frames / 10.125 s / 8 steps**,
`row_stream`, `prefetch`, **CPU merge**, and the `TP cache` path
`/var/tmp/vdn_tp_cache`. It launches `src/inference/infer_tp.py` with
`torch.distributed.run --standalone --nproc_per_node=2`. The first TP task loads the
model once and leaves a **persistent TP worker** running; later tasks with the same
checkpoint/device/offload/TP settings reuse it and only run denoise and decode.
Changing steps, frames, size, seed, or prompt does not restart the worker. Changing
checkpoint, device, sampling, offload, parity, merge, or cache settings does.
Prompt encodings are also cached by mode/prompt/images/model, so repeating a prompt
skips the text-encoder subprocess. The **分块卸载到 CPU** checkbox controls TP:
checked = `render.offload_dit=true`, which streams each rank's sharded blocks from
CPU. The **TP 卸载策略** selector chooses `retained` or `prefetch` (fast preset).
Retained restores unchanged weights from immutable CPU backing without D2H copies;
prefetch pins that backing and copies the next block on a private stream. Prefetch
costs about 36 GB pinned host memory per rank; both ranks pin concurrently, so the
pin step is about 19 s and total prefetch setup about 38 s. Neither option
changes GEMM shapes or collective order. TP still requires `npu_resident_blocks=0`;
the encoder retains its independent offload. `row_stream` is the recommended parity
mode for large renders. `row_gather` also supports block offload. `off` is fastest but
non-bitwise. `TP cache` is a precomputed CPU-merged shard cache and skips CPU assembly;
if the path is absent or its manifest does not match the configured checkpoint, the
WebUI generates it before starting the worker (about 8–11 minutes, once). Clear the
field to assemble at launch instead. `NPU merge` disables the
cache and is not bitwise against CPU merge; switch to CPU merge when strict parity matters.

The original resident results were bitwise at 39f/512x288 for the parity modes.
Resident `row_stream` completed 243f/1280x704 but OOMed at 243f/1920x1088; these are
resident measurements, not a limit on the new offloaded path. The UI deliberately
does not gate canvas/frame count. TP is rejected for CUDA, CUDA FP8, resident-block
overrides, and the separate pinned-offload option. Transfer time can reduce NPU
compute utilization during block onload/offload; the NPU triangular inverse raises
sampled AICore from about 37/32% to 84/87% on 39f/8NFE and 93/94% at 1080p. TP logs
now show loading, denoise step completion and decode progress.

With the offloaded `row_stream` path and the NPU triangular inverse, the bundled T2VA
test at **1920x1088 / 243f / 1NFE** completed the MP4: prefetch denoise 110.6 s,
decode 110.1 s, E2E 286.6 s, with prefetch adding a second 1.38 GB block lease. A retained probe measured
38.69 GB live allocation / 62.49 GB reserved per rank, so physical HBM remained close
to full. These validate completion, not full 8-NFE image quality.
The 243f/1920x1088 synthetic FL2VA shape test (4398 prompt rows and two zero
keyframes) also completed two denoise steps and decoding in 532.1 s. Separately,
39f/512x288/8NFE using the CPU-merged cache and offloaded `row_stream` reproduced
both latents and the MP4 SHA-256 exactly. Large-geometry 8-NFE acceptance is still
separate from those tests; offload itself does not remove NPU-merge numerical drift.
For 39f/4NFE, prefetch reduced denoise 72.3 s → 25.4 s but E2E was slower (149.5 s vs
123.3 s) because of pin setup and decode; it is intended for longer multi-step runs.
At 39f/8NFE the same A/B was retained denoise 140.8 s / E2E 342.8 s versus prefetch
denoise 50.8 s / E2E 167.2 s, with byte-identical latents. The TP path now drops the
transformer and its CPU backing before VAE decode; decode fell 47.9 s → 27.2 s and
prefetch E2E to 148.95 s. For TP, choose `prefetch` in the TP strategy selector when
the extra 36 GB pinned host memory per rank is available; retain the default for
one-step or memory-constrained runs.

TP also exposes the experimental LoRA merge choice. `CPU 合并` keeps the original
bitwise path. `NPU 合并` reads only the rank's base/branch shard, moves 33.7 GiB to the
NPU, and folds the LoRA pairs there; it starts in about 55 s on a warm base cache and
is non-bitwise (39f 8NFE measured about 23.7 dB RGB PSNR / 0.898 SSIM against the
single-card reference). A parity label is not a bitwise claim on this merge path. For the no-cache path with bitwise quality,
select `CPU delta + NPU 更新` (about 206 s setup); the runtime never builds the full
65.7 GiB CPU model and uses no 67 GiB merged cache.

Pinned-host block offload (`kernels.npu_pinned_offload=true`) is CLI-only and **not
exposed in the WebUI**, because it reserves about 97 GiB of host pinned memory for the
65.7 GiB transformer. It is default-off, requires `render.offload_dit=true` and a hybrid
checkpoint, and cannot be combined with NPU resident blocks. At 1344×768 it cut 39-frame
1-NFE denoise from 67.8 s to 17.6 s (bitwise identical) and 345-frame 1-NFE from 214.0 s
to 163.9 s; 39-frame 8-NFE denoise improved from 547.3 s to 142.7 s. Pinning adds about
34 s at startup, so a single low-NFE render may not improve end-to-end wall time. The
loader rejects the option unless host `MemAvailable` covers 1.5× the model bytes plus
8 GiB.

Advanced settings also include a random seed (default 42) and an optional local
MiniMax-H3 text encoder snapshot directory containing `processor/` and `text_encoder/`.
If left blank, those components are fetched from `MiniMaxAI/MiniMax-H3` into
`ckpts/text-encoder/` on first use. A complete local copy is reused without network
access. The approximately 66.7 GB conditioner is separate from the main `ckpts/h3-base`
weights; its files are stored on the project volume rather than under `~/.cache`.

The launcher and worker default to `HF_HUB_DISABLE_XET=1`, using resumable HTTP downloads.
This avoids Xet/CAS `401 Unauthorized` failures seen with some Hub mirrors. An explicit
`HF_HUB_DISABLE_XET=0` opts into Xet. Existing completed files in the download directory
are retained on retry. To download the conditioner before submitting a WebUI task:

```bash
.venv/bin/python -c 'from src.paths import upstream_snapshot; print(upstream_snapshot("processor", "text_encoder"))'
```

## Logs, progress, and files

Submission opens a standalone `/logs/<id>` page. Both that page and the homepage poll
status and the log tail once per second, show stage progress, and offer an MP4 player
and download link on completion. History links remain available on the homepage.

The overall percentage is weighted by stage. Loading and input encoding have stage-based
estimates; denoising reports each completed model evaluation (for example, `3/8`) and
its measured duration via `[vdn-progress]` log events. Decoding starts at 85%; successful
process exit with an output file marks the task done. Merged
stdout/stderr and wrapper start/finish markers are retained in the task log. The log
endpoint returns the latest 256 KiB; the complete log stays on disk.

Each request gets a unique, ignored run directory:

```text
webui/runs/<task-id>/
  inputs/       uploaded images, in their original order
  prompt.txt    original prompt
  prompt.pt     encoder output used by infer.py
  task.log      merged encoder/renderer output and WebUI markers
  state.json    persisted request options, status, progress, errors
  output.mp4    final rendered video and audio
```

Only one task runs at a time; concurrent submissions receive HTTP 409. Subprocesses use
argument lists and the server's `sys.executable`, with repository-root working
directories and absolute script/input/output paths. Failed processes leave an error
state and their logs. Shutdown terminates the supervised process, and interrupted
tasks are marked as errors when history is loaded after a restart; tasks are not
automatically resumed. `WEBUI_RUNS_DIR=/absolute/path` can relocate run storage.

## HTTP API

| Method | Route | Result |
|---|---|---|
| `GET` | `/` | Generation form and latest-task panel |
| `POST` | `/api/tasks` | Multipart form submission; HTTP 202 with `{"id":"..."}` |
| `GET` | `/logs/<id>` | Standalone logs, progress, and result page |
| `GET` | `/api/tasks` | `{"tasks":[...]}` history, newest first |
| `GET` | `/api/tasks/<id>` | JSON task state |
| `GET` | `/api/tasks/<id>/log` | Plain-text log tail |
| `GET` | `/api/tasks/<id>/video` | Completed MP4, including byte-range support |

Form fields: `prompt`, `mode`, `sampling`, `width`, `height`, `duration`, `fps`, `device`,
`checkpoint`, `steps`, `frames` (legacy fallback), `offload`, `optimized_cuda`, `seed`,
`model_root`, `npu_resident_blocks`, `tensor_parallel`, `tp_parity`,
`tp_merge_device`, `tp_merge_delta_device`; file fields:
`first`, `last`, or repeated `refs`. Send explicit
`true`/`false` for checkboxes when using the API. Invalid requests return HTTP 400;
requests exceeding the total size cap return 413. Unknown tasks and unfinished videos
return 404. Task `status` is `queued`, `running`, `done`, or `error`; `stage` progresses
through `queued`, `encoding`, `loading`, `warmup` (when enabled), `denoise`, `decode`,
`done` (or `error`). `denoise_step` / `denoise_total` and `step_seconds` describe the
latest completed denoising evaluation. Task state also includes `actual_duration`,
`output_frames`, and the selected canvas, frame rate, and sampling scheme.

The local-tool API has no authentication. Static assets are allowlisted; runtime file
routes expose only a task's log and completed video, with path traversal rejected.

## Dry run and tests

Exercise uploads, validation, task transitions, logs, and video delivery without model
dependencies or checkpoints:

```bash
WEBUI_DRY_RUN=1 bash scripts/launch_webui.sh
```

Dry-run tasks are labeled in the UI and state. They create **placeholder** cache/video
files, not a valid Torch cache or a playable video. Start the server without that
environment variable for real inference.

Run the standard-library tests and syntax checks from the repository root:

```bash
python -m unittest webui.test_server -v
python -m compileall -q webui
bash -n scripts/launch_webui.sh
```

Tests cover every mode's HTTP flow, mode/image/frame validation, correct command and
config selection, filename preservation, logs, history persistence, video ranges,
concurrent submissions, and subprocess failure reporting. They do not load models.
