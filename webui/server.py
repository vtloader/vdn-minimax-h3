"""Browser UI and task runner for the repository's single-device inference path.

The server deliberately uses only Python's standard library.  The actual encoders and
renderer remain the repository entry points in ``src/inference``; this module validates a
request, stores its inputs under one run directory, and supervises those entry points.

Run from the repository root with::

    python -m webui.server

Set ``WEBUI_DRY_RUN=1`` to exercise the HTTP API and UI without loading a model.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import unquote, urlparse

from src.inference.geometry import (DEFAULT_HEIGHT, DEFAULT_WIDTH, EXPORT_FPS,
                                    NATIVE_FPS, export_frame_count, frames_for_duration,
                                    validate_canvas)


REPO_ROOT = Path(__file__).resolve().parent.parent
WEBUI_ROOT = REPO_ROOT / "webui"
STATIC_ROOT = WEBUI_ROOT / "static"
RUNS_DIR = Path(os.environ.get("WEBUI_RUNS_DIR", str(WEBUI_ROOT / "runs"))).resolve()

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_REQUEST_BYTES = 80 * 1024 * 1024
MAX_PROMPT_LENGTH = 20_000
MAX_REFERENCE_IMAGES = 16
MAX_FRAMES = 4096
MAX_STEPS = 100
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TASK_ID_PATTERN = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
DEVICE_PATTERN = re.compile(r"^(?:npu|cuda):[0-9]+$")
ACTIVE_STATUSES = {"queued", "running"}

ENCODE_PROMPT = REPO_ROOT / "src" / "inference" / "encode_prompt.py"
ENCODE_KEYFRAMES = REPO_ROOT / "src" / "inference" / "encode_keyframes.py"
INFER = REPO_ROOT / "src" / "inference" / "infer.py"
INFER_TP = REPO_ROOT / "src" / "inference" / "infer_tp.py"
CONFIG_BF16 = REPO_ROOT / "configs" / "inference" / "8nfe.yaml"
CONFIG_CUDA_FP8 = REPO_ROOT / "configs" / "inference" / "8nfe_tuned_fp8.yaml"
CONFIG_STANDARD = REPO_ROOT / "configs" / "inference" / "50nfe.yaml"
CONFIG_STANDARD_FP8 = REPO_ROOT / "configs" / "inference" / "50nfe_tuned_fp8.yaml"
SAMPLING_CHECKPOINTS = {"accelerated": "ckpts/stage-dmd-step-250", "standard": "ckpts/stage-b-step-2000"}
TP_PARITIES = {"off", "row_gather", "row_stream"}
TP_OFFLOAD_STRATEGIES = {"retained", "prefetch"}
TP_MERGE_DEVICES = {"cpu", "npu"}
TP_MERGE_DELTA_DEVICES = {"npu", "cpu"}
TP_WORKER_QUEUE = ".tp_worker"
TP_WORKER_READY_TIMEOUT = 900
TP_WORKER_REQUEST_TIMEOUT = 3600


class SubmissionError(ValueError):
    """A user-controlled request failed validation."""


class TaskConflictError(RuntimeError):
    """Another task is currently running."""


@dataclass(frozen=True)
class UploadedFile:
    field: str
    filename: str
    content: bytes
    content_type: str = ""


@dataclass(frozen=True)
class Submission:
    mode: str
    prompt: str
    device: str
    checkpoint: str
    steps: int
    frames: int
    offload: bool
    optimized_cuda: bool
    uploads: dict[str, list[UploadedFile]]
    seed: int = 42
    model_root: str = ""
    sampling: str = "accelerated"
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = NATIVE_FPS
    duration: float | None = None
    npu_resident_blocks: int = 0
    tensor_parallel: bool = False
    tp_parity: str = "row_stream"
    tp_offload_strategy: str = "retained"
    tp_cache: str = ""
    tp_merge_device: str = "cpu"
    tp_merge_delta_device: str = "npu"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _one(fields: Mapping[str, list[str]], name: str, default: str = "") -> str:
    values = fields.get(name, [])
    if len(values) > 1:
        raise SubmissionError(f"{name} must be provided only once")
    return values[0] if values else default


def _parse_integer(value: str, name: str, minimum: int, maximum: int) -> int:
    if not re.fullmatch(r"[0-9]+", value or ""):
        raise SubmissionError(f"{name} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise SubmissionError(f"{name} must be between {minimum} and {maximum}")
    return number


def _parse_checkbox(fields: Mapping[str, list[str]], name: str, default: bool) -> bool:
    value = _one(fields, name, "true" if default else "false").strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise SubmissionError(f"{name} must be true or false")
    return value in {"1", "true", "yes", "on"}


def _safe_filename(filename: str) -> str:
    """Return a single safe path component while retaining the uploaded basename."""
    filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    # Keep ordinary spaces and Unicode characters so the saved file remains recognizable.
    # Strip control characters and characters that can be interpreted as path syntax on
    # common host platforms; the basename split above removes traversal components.
    filename = re.sub(r"[\x00-\x1f\x7f]", "_", filename)
    filename = re.sub(r"[<>:\"|?*]", "_", filename).strip()
    if filename in {"", ".", ".."}:
        raise SubmissionError("uploaded image must have a filename")
    # Filesystem limits are in bytes, including for multibyte Unicode basenames.
    if len(filename.encode("utf-8")) > 180:
        suffix = Path(filename).suffix
        if len(suffix) > 10:
            suffix = ""
        stem = filename[: -len(suffix)] if suffix else filename
        stem = stem.encode("utf-8")[: 180 - len(suffix.encode("utf-8"))].decode("utf-8", errors="ignore")
        filename = stem + suffix
    return filename


def _validate_uploads(uploads: Mapping[str, list[UploadedFile]]) -> None:
    for files in uploads.values():
        for uploaded in files:
            safe_name = _safe_filename(uploaded.filename)
            if Path(safe_name).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                allowed = ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
                raise SubmissionError(f"{uploaded.filename!r} is not a supported image ({allowed})")
            if not uploaded.content:
                raise SubmissionError(f"uploaded image {uploaded.filename!r} is empty")
            if len(uploaded.content) > MAX_IMAGE_BYTES:
                raise SubmissionError(
                    f"uploaded image {uploaded.filename!r} exceeds the 20 MiB limit"
                )
            # Basic content checks without importing Pillow or the model dependencies.
            # The existing encoder performs full image decoding before loading the VLM.
            data = uploaded.content
            suffix = Path(safe_name).suffix.lower()
            valid_header = {
                ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
                ".jpg": data.startswith(b"\xff\xd8\xff"),
                ".jpeg": data.startswith(b"\xff\xd8\xff"),
                ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
                ".bmp": data.startswith(b"BM"),
            }
            if not valid_header[suffix]:
                raise SubmissionError(f"{uploaded.filename!r} has invalid image content")


def validate_submission(
    fields: Mapping[str, list[str]], uploads: Mapping[str, list[UploadedFile]]
) -> Submission:
    """Validate form data and return the command-safe request description.

    This helper contains no HTTP or subprocess code, which makes the mode and argument
    rules easy to test without importing the model stack.
    """
    mode = _one(fields, "mode", "t2va").strip().lower()
    if mode not in {"t2va", "i2va", "fl2va", "ref2va"}:
        raise SubmissionError("mode must be t2va, i2va, fl2va, or ref2va")

    prompt = _one(fields, "prompt")
    if not prompt.strip() or "\x00" in prompt:
        raise SubmissionError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise SubmissionError("prompt is limited to 20,000 characters")

    device = _one(fields, "device", "npu:0").strip().lower()
    if not DEVICE_PATTERN.fullmatch(device):
        raise SubmissionError("device must match npu:0 or cuda:0 (with a numeric index)")

    sampling = _one(fields, "sampling", "").strip().lower()
    if sampling and sampling not in {"accelerated", "standard", "custom"}:
        raise SubmissionError("sampling must be accelerated, standard, or custom")
    checkpoint = _one(fields, "checkpoint", SAMPLING_CHECKPOINTS.get(sampling or "accelerated", "")).strip()
    if (not checkpoint or len(checkpoint) > 1024 or "${" in checkpoint
            or any(ord(character) < 32 for character in checkpoint)):
        raise SubmissionError("checkpoint must be a valid non-empty path")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    checkpoint = str(checkpoint_path.resolve())
    builtin_paths = {key: str((REPO_ROOT / path).resolve()) for key, path in SAMPLING_CHECKPOINTS.items()}
    if not sampling:
        sampling = next((key for key, path in builtin_paths.items() if path == checkpoint), "custom")
    if sampling in builtin_paths and checkpoint != builtin_paths[sampling]:
        raise SubmissionError("sampling preset and checkpoint differ; select custom for a custom checkpoint")
    steps = _parse_integer(_one(fields, "steps", "50" if sampling == "standard" else "8"), "steps", 1, MAX_STEPS)
    width = _parse_integer(_one(fields, "width", str(DEFAULT_WIDTH)), "width", 256, 2048)
    height = _parse_integer(_one(fields, "height", str(DEFAULT_HEIGHT)), "height", 256, 2048)
    try:
        validate_canvas(width, height)
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc
    fps = _parse_integer(_one(fields, "fps", "24"), "fps", 1, 120)
    if fps not in EXPORT_FPS:
        raise SubmissionError(f"fps must be one of {EXPORT_FPS}")
    duration_value = _one(fields, "duration", "").strip()
    duration = None
    if duration_value:
        try:
            duration = float(duration_value)
        except ValueError as exc:
            raise SubmissionError("duration must be a number in seconds") from exc
        if not math.isfinite(duration) or not 1 <= duration <= 15:
            raise SubmissionError("duration must be between 1 and 15 seconds")
        frames = frames_for_duration(duration)
    else:
        frames = _parse_integer(_one(fields, "frames", "345"), "frames", 6, MAX_FRAMES)
        if (frames - 5) % 17 != 0:
            raise SubmissionError("frames must be aligned as 17*n+5 (for example, 345)")
    seed = _parse_integer(_one(fields, "seed", "42"), "seed", 0, 2**63 - 1)
    model_root = _one(fields, "model_root", "").strip()
    if model_root:
        if len(model_root) > 1024 or any(ord(c) < 32 for c in model_root):
            raise SubmissionError("model_root must be a local snapshot directory")
        snapshot = Path(model_root).expanduser()
        if not snapshot.is_absolute():
            snapshot = REPO_ROOT / snapshot
        model_root = str(snapshot.resolve())

    if any(name not in {"first", "last", "refs", "refs[]"} for name in uploads):
        raise SubmissionError("unknown image upload field")
    normalized_uploads: dict[str, list[UploadedFile]] = {
        "first": list(uploads.get("first", [])),
        "last": list(uploads.get("last", [])),
        "refs": list(uploads.get("refs", [])) + list(uploads.get("refs[]", [])),
    }
    _validate_uploads(normalized_uploads)

    if mode == "t2va":
        if any(normalized_uploads.values()):
            raise SubmissionError("t2va does not accept image uploads")
    elif mode == "i2va":
        if len(normalized_uploads["first"]) != 1 or any(
            normalized_uploads[name] for name in ("last", "refs")
        ):
            raise SubmissionError("i2va requires exactly one first image")
    elif mode == "fl2va":
        if len(normalized_uploads["first"]) != 1 or len(normalized_uploads["last"]) != 1:
            raise SubmissionError("fl2va requires exactly one first and one last image")
        if normalized_uploads["refs"]:
            raise SubmissionError("fl2va does not accept reference images")
    elif mode == "ref2va":
        if not 1 <= len(normalized_uploads["refs"]) <= MAX_REFERENCE_IMAGES:
            raise SubmissionError(
                f"ref2va requires between 1 and {MAX_REFERENCE_IMAGES} reference images"
            )
        if normalized_uploads["first"] or normalized_uploads["last"]:
            raise SubmissionError("ref2va accepts reference images through the refs field")

    offload = _parse_checkbox(fields, "offload", device.startswith("npu:"))
    optimized_cuda = _parse_checkbox(fields, "optimized_cuda", False)
    if optimized_cuda and not device.startswith("cuda:"):
        raise SubmissionError("CUDA optimized FP8 can only be used with a CUDA device")
    if optimized_cuda and offload:
        raise SubmissionError("CUDA FP8 does not support CPU offload; disable one of them")
    resident = _parse_integer(_one(fields, "npu_resident_blocks", "0"),
                              "npu_resident_blocks", 0, 49)
    if resident:
        if not device.startswith("npu:"):
            raise SubmissionError("NPU resident blocks can only be used with an NPU device")
        if not offload:
            raise SubmissionError("NPU resident blocks require CPU block offload")

    tensor_parallel = _parse_checkbox(fields, "tensor_parallel", False)
    tp_parity = _one(fields, "tp_parity", "row_stream").strip().lower()
    if tp_parity not in TP_PARITIES:
        raise SubmissionError("tp_parity must be off, row_gather or row_stream")
    tp_offload_strategy = _one(fields, "tp_offload_strategy", "retained").strip().lower()
    if tp_offload_strategy not in TP_OFFLOAD_STRATEGIES:
        raise SubmissionError("tp_offload_strategy must be retained or prefetch")
    if tp_offload_strategy == "prefetch" and (not tensor_parallel or not offload):
        raise SubmissionError("tp_offload_strategy=prefetch requires tensor_parallel=true and offload=true")
    tp_merge_device = _one(fields, "tp_merge_device", "cpu").strip().lower()
    if tp_merge_device not in TP_MERGE_DEVICES:
        raise SubmissionError("tp_merge_device must be cpu or npu")
    tp_merge_delta_device = _one(fields, "tp_merge_delta_device", "npu").strip().lower()
    if tp_merge_delta_device not in TP_MERGE_DELTA_DEVICES:
        raise SubmissionError("tp_merge_delta_device must be npu or cpu")
    tp_cache = _one(fields, "tp_cache", "").strip()
    if (len(tp_cache) > 1024 or "\x00" in tp_cache or "${" in tp_cache
            or any(ord(character) < 32 for character in tp_cache)):
        raise SubmissionError("tp_cache must be a valid path")
    if tp_cache and not tensor_parallel:
        raise SubmissionError("tp_cache requires tensor_parallel=true")
    if tp_cache and tp_merge_device == "npu":
        raise SubmissionError("tp_cache cannot be combined with tp_merge_device=npu")
    if not tensor_parallel and (tp_merge_device != "cpu" or tp_merge_delta_device != "npu"):
        raise SubmissionError("TP merge options require tensor_parallel=true")
    if tensor_parallel:
        if not device.startswith("npu:"):
            raise SubmissionError("experimental tensor parallel can only be used with NPU")
        if optimized_cuda:
            raise SubmissionError("experimental tensor parallel does not support CUDA FP8")
        if resident:
            raise SubmissionError("experimental tensor parallel requires npu_resident_blocks=0")
        if tp_merge_device == "cpu" and tp_merge_delta_device != "npu":
            raise SubmissionError("tp_merge_delta_device is only valid with tp_merge_device=npu")

    return Submission(
        mode=mode,
        prompt=prompt,
        device=device,
        checkpoint=checkpoint,
        steps=steps,
        frames=frames,
        offload=offload,
        optimized_cuda=optimized_cuda,
        uploads=normalized_uploads,
        seed=seed,
        model_root=model_root,
        sampling=sampling,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        npu_resident_blocks=resident,
        tensor_parallel=tensor_parallel,
        tp_parity=tp_parity,
        tp_offload_strategy=tp_offload_strategy,
        tp_cache=tp_cache,
        tp_merge_device=tp_merge_device,
        tp_merge_delta_device=tp_merge_delta_device,
    )


def _relative_input_name(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


def _unique_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for index in range(2, 10_000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise SubmissionError("too many uploaded files with the same name")


def inference_config_path(task: Mapping[str, object]) -> Path:
    """Resolve the inference YAML for a task (sampling and CUDA FP8 both select it)."""
    standard = task.get("sampling") == "standard"
    if str(task["device"]).startswith("cuda:") and bool(task.get("optimized_cuda")):
        return CONFIG_STANDARD_FP8 if standard else CONFIG_CUDA_FP8
    return CONFIG_STANDARD if standard else CONFIG_BF16


def build_commands(task: Mapping[str, object], run_dir: Path) -> list[list[str]]:
    """Build the encoder and renderer argv lists for a validated task."""
    python = sys.executable
    mode = str(task["mode"])
    cache = str(run_dir / "prompt.pt")
    input_paths = task["input_paths"]
    assert isinstance(input_paths, dict)

    if mode == "t2va":
        encoder = [
            python,
            "-u",
            str(ENCODE_PROMPT),
            f"--prompt={task['prompt']}",
            "--out",
            cache,
            "--device",
            str(task["device"]),
        ]
    else:
        encoder = [
            python,
            "-u",
            str(ENCODE_KEYFRAMES),
            f"--prompt={task['prompt']}",
            "--out",
            cache,
            "--device",
            str(task["device"]),
        ]
    if task.get("model_root"):
        encoder.extend(["--model_root", str(task["model_root"])])
    if task["offload"] or task.get("tensor_parallel"):
        encoder.append("--offload")
    if mode != "t2va":
        encoder.extend(["--width", str(task.get("width", DEFAULT_WIDTH)),
                        "--height", str(task.get("height", DEFAULT_HEIGHT))])
        if mode == "i2va":
            encoder.extend(["--first", str(input_paths["first"][0])])
        elif mode == "fl2va":
            encoder.extend(
                ["--first", str(input_paths["first"][0]), "--last", str(input_paths["last"][0])]
            )
        elif mode == "ref2va":
            encoder.append("--refs")
            encoder.extend(str(path) for path in input_paths["refs"])

    config = inference_config_path(task)
    tensor_parallel = bool(task.get("tensor_parallel"))
    infer = [python, "-u"]
    if tensor_parallel:
        infer.extend(["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                      "--nproc_per_node=2", str(INFER_TP)])
    else:
        infer.append(str(INFER))
    infer.extend([
        "--config",
        str(config),
        f"checkpoint={json.dumps(task['checkpoint'], ensure_ascii=False)}",
        f"render.prompt_file={json.dumps(cache, ensure_ascii=False)}",
        f"render.out={json.dumps(str(run_dir / 'output.mp4'), ensure_ascii=False)}",
        f"render.device={task['device']}",
        f"render.num_steps={task['steps']}",
        f"render.num_frames={task['frames']}",
        f"render.offload_dit={'true' if task['offload'] else 'false'}",
        f"render.seed={task.get('seed', 42)}",
        f"render.width={task.get('width', DEFAULT_WIDTH)}",
        f"render.height={task.get('height', DEFAULT_HEIGHT)}",
        f"render.fps={task.get('fps', NATIVE_FPS)}",
        f"kernels.npu_resident_blocks={0 if tensor_parallel else task.get('npu_resident_blocks', 0)}",
    ])
    if tensor_parallel:
        infer.extend(["parallel.npu_tensor_parallel=true",
                      f"parallel.npu_tp_parity={task.get('tp_parity', 'row_stream')}",
                      f"parallel.merge_device={task.get('tp_merge_device', 'cpu')}"])
        if task.get("offload"):
            infer.append(f"parallel.tp_offload_strategy="
                         f"{task.get('tp_offload_strategy', 'retained')}")
        if task.get("tp_cache"):
            infer.append(f"parallel.tp_cache={json.dumps(task['tp_cache'], ensure_ascii=False)}")
        if task.get("tp_merge_device") == "npu":
            infer.append(f"parallel.merge_delta_device="
                         f"{task.get('tp_merge_delta_device', 'npu')}")
    return [encoder, infer]


class TaskManager:
    """Own run directories, one active subprocess chain, state, and history."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, object]] = {}
        self._active_id: str | None = None
        self._process: subprocess.Popen | None = None
        self._worker: threading.Thread | None = None
        self._tp_worker_process: subprocess.Popen | None = None
        self._tp_worker_reader: threading.Thread | None = None
        self._tp_worker_signature: dict[str, object] | None = None
        self._tp_worker_task_id: str | None = None
        self._tp_worker_queue = self.runs_dir / TP_WORKER_QUEUE
        self._tp_worker_log_path = self._tp_worker_queue / "worker.log"
        self._prompt_cache = self.runs_dir / ".prompt_cache"
        self._prompt_cache.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self.dry_run = os.environ.get("WEBUI_DRY_RUN", "").strip() == "1"
        self._load_history()

    def _load_history(self) -> None:
        for state_path in sorted(self.runs_dir.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    continue
                task_id = str(state.get("id", ""))
                if (not TASK_ID_PATTERN.fullmatch(task_id) or state_path.parent.name != task_id
                        or state_path.resolve().parent != self._run_dir(task_id)):
                    continue
                if state.get("status") in ACTIVE_STATUSES:
                    state["status"] = "error"
                    state["stage"] = "error"
                    state["error"] = "server restarted while task was active"
                    state["updated_at"] = utc_now()
                    self._atomic_write(state_path, state)
                self._tasks[task_id] = state
            except (OSError, ValueError, TypeError, KeyError):
                continue

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _run_dir(self, task_id: str) -> Path:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise KeyError(task_id)
        path = self.runs_dir / task_id
        if path.resolve() != path:
            raise KeyError(task_id)
        return path

    def _artifact_path(self, task_id: str, filename: str) -> Path:
        run_dir = self._run_dir(task_id)
        path = run_dir / filename
        if not path.resolve().is_relative_to(run_dir):
            raise KeyError(task_id)
        return path

    def _persist_unlocked(self, task: Mapping[str, object]) -> None:
        self._atomic_write(self._run_dir(str(task["id"])) / "state.json", task)

    def _update(self, task_id: str, **updates: object) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.update(updates)
            task["updated_at"] = utc_now()
            self._persist_unlocked(task)

    def _append_log(self, task_id: str, line: str) -> None:
        run_dir = self._run_dir(task_id)
        with (run_dir / "task.log").open("a", encoding="utf-8") as log_file:
            log_file.write(line.rstrip("\n") + "\n")
            log_file.flush()

    def _record_output(self, task_id: str, line: str) -> None:
        line = line.rstrip("\n")
        self._append_log(task_id, line)
        if line.startswith("[vdn-progress] "):
            try:
                event = json.loads(line[len("[vdn-progress] "):])
                stage = event["stage"]
                updates: dict[str, object] = {"stage": stage}
                if stage == "denoise":
                    total, step = event["total"], event["completed"]
                    if (type(total) is not int or type(step) is not int
                            or not 1 <= total <= MAX_STEPS or not 0 <= step <= total):
                        return
                    updates.update(progress=60 + int(25 * step / total),
                                   denoise_step=step, denoise_total=total)
                    if isinstance(event.get("seconds"), (int, float)):
                        updates["step_seconds"] = event["seconds"]
                elif stage in {"loading", "warmup", "decode"}:
                    updates["progress"] = {"loading": 50, "warmup": 58, "decode": 85}[stage]
                else:
                    return
                updates["progress"] = max(self.progress(task_id), updates["progress"])
                self._update(task_id, **updates)
                return
            except (ValueError, TypeError, KeyError):
                pass
        lower = line.lower()
        with self._lock:
            stage = self._tasks[task_id]["stage"]
        if stage == "encoding":
            if "wrote" in lower and ".pt" in lower:
                self._update(task_id, progress=max(self.progress(task_id), 45))
            elif "encoding" in lower:
                self._update(task_id, progress=max(self.progress(task_id), 20))
        elif "timing: denoise" in lower:
            self._update(task_id, stage="decode", progress=max(self.progress(task_id), 85))
            self._append_log(task_id, "[webui] denoise complete; decode started")
        elif "warming" in lower and stage == "denoise":
            self._update(task_id, progress=max(self.progress(task_id), 60))
        elif "wrote" in lower and ".mp4" in lower:
            self._update(task_id, stage="decode", progress=max(self.progress(task_id), 95))

    def progress(self, task_id: str) -> int:
        with self._lock:
            return int(self._tasks[task_id].get("progress", 0))

    def _public_task(self, task: Mapping[str, object]) -> dict[str, object]:
        result = dict(task)
        result.pop("input_paths", None)
        result.pop("prompt", None)
        return result

    def get(self, task_id: str) -> dict[str, object] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else self._public_task(task)

    def history(self) -> list[dict[str, object]]:
        with self._lock:
            tasks = [self._public_task(task) for task in self._tasks.values()]
        tasks.sort(key=lambda task: str(task.get("created_at", "")), reverse=True)
        return tasks

    def read_log(self, task_id: str, max_bytes: int = 256 * 1024) -> str:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        path = self._artifact_path(task_id, "task.log")
        try:
            with path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - max_bytes))
                data = log_file.read()
            return data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def video_path(self, task_id: str) -> Path | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.get("status") != "done":
                return None
        try:
            path = self._artifact_path(task_id, "output.mp4")
        except KeyError:
            return None
        return path if path.is_file() else None

    def submit(self, submission: Submission) -> str:
        with self._lock:
            if self._closed:
                raise TaskConflictError("server is stopping")
            if self._active_id is not None:
                raise TaskConflictError("only one inference task can run at a time")

            task_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = self._run_dir(task_id)
            input_dir = run_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "prompt.txt").write_text(submission.prompt, encoding="utf-8")

            input_paths: dict[str, list[str]] = {"first": [], "last": [], "refs": []}
            input_files: list[str] = []
            for field in ("first", "last", "refs"):
                for uploaded in submission.uploads[field]:
                    filename = _safe_filename(uploaded.filename)
                    destination = _unique_path(input_dir, filename)
                    destination.write_bytes(uploaded.content)
                    path_string = str(destination)
                    input_paths[field].append(path_string)
                    input_files.append(_relative_input_name(destination, run_dir))

            task: dict[str, object] = {
                "id": task_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "mode": submission.mode,
                "prompt": submission.prompt,
                "device": submission.device,
                "checkpoint": submission.checkpoint,
                "steps": submission.steps,
                "frames": submission.frames,
                "offload": submission.offload,
                "optimized_cuda": submission.optimized_cuda,
                "seed": submission.seed,
                "model_root": submission.model_root,
                "sampling": submission.sampling,
                "width": submission.width,
                "height": submission.height,
                "fps": submission.fps,
                "duration": submission.duration,
                "npu_resident_blocks": submission.npu_resident_blocks,
                "tensor_parallel": submission.tensor_parallel,
                "tp_parity": submission.tp_parity,
                "tp_offload_strategy": submission.tp_offload_strategy,
                "tp_cache": submission.tp_cache,
                "tp_merge_device": submission.tp_merge_device,
                "tp_merge_delta_device": submission.tp_merge_delta_device,
                "actual_duration": submission.frames / NATIVE_FPS,
                "output_frames": export_frame_count(submission.frames, submission.fps),
                "dry_run": self.dry_run,
                "input_files": input_files,
                "input_paths": input_paths,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "error": None,
            }
            self._tasks[task_id] = task
            self._active_id = task_id
            self._persist_unlocked(task)
            self._worker = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
            self._worker.start()
            return task_id

    def _set_marker(self, task_id: str, stage: str, progress: int, message: str) -> None:
        self._update(task_id, status="running", stage=stage, progress=progress)
        self._append_log(task_id, f"[webui] {message}")

    def _run_dry_stage(self, task_id: str, command: list[str], stage: str) -> None:
        self._append_log(task_id, f"[dry-run] {shlex.join(command)}")
        if stage == "encoding":
            self._record_output(task_id, "encoding: dry-run encoder")
            self._record_output(task_id, f"wrote {self._run_dir(task_id) / 'prompt.pt'}")
            (self._run_dir(task_id) / "prompt.pt").write_bytes(b"VDN-H3 WEBUI DRY RUN PROMPT CACHE\n")
        else:
            self._record_output(task_id, '[vdn-progress] {"stage":"loading"}')
            steps = int(self._tasks[task_id]["steps"])
            for step in range(steps + 1):
                self._record_output(task_id, "[vdn-progress] " + json.dumps({
                    "stage": "denoise", "completed": step, "total": steps, "seconds": 0.01,
                }))
            self._record_output(task_id, "timing: denoise 0.01s (0.01s/NFE)")
            self._record_output(task_id, f"wrote {self._run_dir(task_id) / 'output.mp4'}")
            (self._run_dir(task_id) / "output.mp4").write_bytes(
                b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom\x00\x00\x00\x08free"
            )
        time.sleep(0.02)

    def _execute(self, task_id: str, command: list[str]) -> None:
        self._append_log(task_id, f"[webui] exec: {shlex.join(command)}")
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("HF_HUB_DISABLE_XET", "1")
        with self._lock:
            if self._closed:
                raise RuntimeError("server stopped while task was active")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                raise RuntimeError(f"could not start subprocess: {exc}") from exc
            self._process = process

        assert process.stdout is not None

        reader_errors: list[Exception] = []

        def read_lines() -> None:
            try:
                with process.stdout:
                    for line in process.stdout:
                        self._record_output(task_id, line)
            except Exception as exc:
                reader_errors.append(exc)
                process.terminate()

        reader = threading.Thread(target=read_lines, name=f"webui-log-{task_id}", daemon=True)
        reader.start()
        return_code = process.wait()
        reader.join()
        with self._lock:
            self._process = None
        if reader_errors:
            raise RuntimeError(f"could not read subprocess output: {reader_errors[0]}")
        if return_code != 0:
            raise RuntimeError(f"subprocess exited with code {return_code}")

    def _prompt_key(self, task: Mapping[str, object]) -> str:
        digest = hashlib.sha256()
        for value in (task.get("mode"), task.get("prompt"), task.get("model_root"),
                      task.get("device"), task.get("width"), task.get("height")):
            digest.update(str(value or "").encode("utf-8", errors="replace"))
            digest.update(b"\0")
        for field in ("first", "last", "refs"):
            paths = (task.get("input_paths") or {}).get(field) or []
            for path in paths:
                digest.update(field.encode("ascii"))
                digest.update(b"\0")
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
        return digest.hexdigest()[:32]

    @staticmethod
    def _tp_cache_manifest_usable(manifest: object, cache_path: Path,
                                  checkpoint: str) -> bool:
        """Cheap structural check; the worker still validates the full manifest."""
        if not isinstance(manifest, dict):
            return False
        if manifest.get("format_version") != 1 or manifest.get("kind") != "vdn_tp_shard":
            return False
        if manifest.get("world_size") != 2 or manifest.get("checkpoint") != checkpoint:
            return False
        files = manifest.get("rank_files") or {}
        sizes = manifest.get("rank_file_bytes") or {}
        for rank in ("0", "1"):
            name = files.get(rank)
            if not isinstance(name, str) or not name or os.path.basename(name) != name:
                return False
            path = cache_path / name
            if not path.is_file():
                return False
            expected = sizes.get(rank)
            if expected is not None and path.stat().st_size != int(expected):
                return False
        return True

    def _ensure_tp_cache(self, task_id: str, task: Mapping[str, object]) -> None:
        """Build a missing or stale TP cache before the worker starts."""
        cache = str(task.get("tp_cache") or "").strip()
        if not cache or self.dry_run:
            return
        cache_path = Path(cache).expanduser()
        if not cache_path.is_absolute():
            cache_path = REPO_ROOT / cache_path
        manifest_path = cache_path / "manifest.json"
        checkpoint = os.path.abspath(str(task["checkpoint"]))
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
            if self._tp_cache_manifest_usable(manifest, cache_path, checkpoint):
                return
        force = cache_path.exists() and any(cache_path.iterdir())
        command = [sys.executable, "-u", "-m", "src.inference.prepare_tp_cache",
                   "--out", str(cache_path)]
        if force:
            command.append("--force")
        command.extend(["--config", str(inference_config_path(task)),
                        f"checkpoint={json.dumps(str(task['checkpoint']), ensure_ascii=False)}"])
        self._set_marker(task_id, "loading", 50,
                         f"TP cache missing or stale; generating {cache_path} "
                         f"(this can take several minutes)")
        self._execute(task_id, command)
        if not (cache_path / "manifest.json").is_file():
            raise RuntimeError("TP cache generation finished without a manifest")

    @staticmethod
    def _tp_signature(task: Mapping[str, object]) -> dict[str, object]:
        keys = ("device", "checkpoint", "sampling", "offload", "tensor_parallel",
                "tp_parity", "tp_offload_strategy", "tp_cache", "tp_merge_device",
                "tp_merge_delta_device", "optimized_cuda", "npu_resident_blocks")
        return {key: task.get(key) for key in keys}

    def _worker_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + existing if existing else "")
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("HF_HUB_DISABLE_XET", "1")
        environment["VDN_TP_WORKER_QUEUE"] = str(self._tp_worker_queue)
        return environment

    def _read_tp_worker_output(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        try:
            with process.stdout:
                for line in process.stdout:
                    with self._lock:
                        task_id = self._tp_worker_task_id
                    if task_id is not None:
                        self._record_output(task_id, line)
                    else:
                        self._tp_worker_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with self._tp_worker_log_path.open("a", encoding="utf-8") as log:
                            log.write(line)
                            log.flush()
        except Exception as exc:
            with self._lock:
                task_id = self._tp_worker_task_id
            if task_id is not None:
                self._append_log(task_id, f"[webui] TP worker log read failed: {exc}")

    def _stop_tp_worker(self) -> None:
        with self._lock:
            process = self._tp_worker_process
            self._tp_worker_process = None
            self._tp_worker_signature = None
        if process is not None and process.poll() is None:
            try:
                (self._tp_worker_queue / "stop").write_text("stop\n", encoding="utf-8")
            except OSError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        reader = self._tp_worker_reader
        if reader is not None:
            reader.join(timeout=5)
        self._tp_worker_reader = None
        with self._lock:
            if self._process is process:
                self._process = None

    def _kill_stale_tp_worker(self) -> None:
        """Reap a worker left behind when a previous server was killed hard."""
        ready = self._tp_worker_queue / "ready"
        if not ready.is_file():
            return
        try:
            pid = int(json.loads(ready.read_text(encoding="utf-8")).get("pid"))
        except (OSError, ValueError, TypeError, AttributeError):
            return
        proc = Path(f"/proc/{pid}")
        try:
            cmdline = proc.joinpath("cmdline").read_bytes().decode("utf-8", "replace")
            stat = proc.joinpath("stat").read_text(encoding="utf-8").split()
            parent = int(stat[3])
        except (OSError, ValueError, IndexError):
            return
        if "infer_tp.py" not in cmdline:
            return
        for target in (pid, parent):
            try:
                os.kill(target, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    def _start_tp_worker(self, task: Mapping[str, object], run_dir: Path) -> None:
        signature = self._tp_signature(task)
        with self._lock:
            process = self._tp_worker_process
            if (process is not None and process.poll() is None
                    and self._tp_worker_signature == signature):
                return
        self._stop_tp_worker()
        self._kill_stale_tp_worker()

        queue = self._tp_worker_queue
        queue.mkdir(parents=True, exist_ok=True)
        for name in ("request.json", "result.json", "stop", "ready"):
            try:
                (queue / name).unlink()
            except FileNotFoundError:
                pass
        self._append_log(str(task["id"]), "[webui] starting persistent TP worker")
        command = build_commands(task, run_dir)[1]
        with self._lock:
            if self._closed:
                raise RuntimeError("server stopped while task was active")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=self._worker_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                raise RuntimeError(f"could not start TP worker: {exc}") from exc
            self._process = process
            self._tp_worker_process = process
            self._tp_worker_signature = signature
        reader = threading.Thread(target=self._read_tp_worker_output, args=(process,),
                                  name="webui-tp-worker-log", daemon=True)
        reader.start()
        self._tp_worker_reader = reader

        ready = queue / "ready"
        deadline = time.monotonic() + TP_WORKER_READY_TIMEOUT
        while not ready.is_file():
            if process.poll() is not None:
                self._stop_tp_worker()
                raise RuntimeError("TP worker exited during model load; see worker.log")
            if time.monotonic() >= deadline:
                self._stop_tp_worker()
                raise RuntimeError("TP worker did not become ready within 15 minutes")
            time.sleep(0.2)

    def _run_tp_worker_task(self, task_id: str, task: Mapping[str, object],
                            run_dir: Path) -> None:
        self._start_tp_worker(task, run_dir)
        request = {
            "prompt_file": str(run_dir / "prompt.pt"),
            "out": str(run_dir / "output.mp4"),
            "num_steps": int(task["steps"]),
            "num_frames": int(task["frames"]),
            "width": int(task["width"]),
            "height": int(task["height"]),
            "fps": int(task["fps"]),
            "seed": int(task.get("seed", 42)),
            "warmup_steps": 0,
            "save_latents": False,
            "latents_root": str(run_dir),
            "record": True,
        }
        queue = self._tp_worker_queue
        result_path = queue / "result.json"
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
        request_path = queue / "request.json"
        temporary = queue / f"request.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
        with self._lock:
            self._tp_worker_task_id = task_id
            self._append_log(task_id, "[webui] submitting request to persistent TP worker")
        os.replace(temporary, request_path)
        deadline = time.monotonic() + TP_WORKER_REQUEST_TIMEOUT
        try:
            while True:
                if result_path.is_file():
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    result_path.unlink()
                    break
                process = self._tp_worker_process
                if process is None or process.poll() is not None:
                    raise RuntimeError("TP worker exited before returning a result; see worker.log")
                if time.monotonic() >= deadline:
                    self._stop_tp_worker()
                    raise RuntimeError("TP worker request timed out")
                time.sleep(0.2)
        finally:
            with self._lock:
                if self._tp_worker_task_id == task_id:
                    self._tp_worker_task_id = None
        if payload.get("status") != "ok":
            raise RuntimeError(payload.get("error") or "TP worker request failed")
        self._append_log(task_id, f"[webui] TP worker timings: {json.dumps(payload.get('timings') or {})}")

    def _run_task(self, task_id: str) -> None:
        run_dir = self._run_dir(task_id)
        try:
            task = self._tasks[task_id]
            commands = build_commands(task, run_dir)
            prompt_path = run_dir / "prompt.pt"
            prompt_key = self._prompt_key(task)
            cached_prompt = self._prompt_cache / f"{prompt_key}.pt"
            self._set_marker(task_id, "encoding", 10, "encoding started")
            if not self.dry_run and cached_prompt.is_file():
                shutil.copyfile(cached_prompt, prompt_path)
                self._append_log(task_id, f"[webui] prompt cache hit: {cached_prompt}")
                self._set_marker(task_id, "encoding", 45, "prompt cache hit")
            else:
                if self.dry_run:
                    self._run_dry_stage(task_id, commands[0], "encoding")
                else:
                    self._execute(task_id, commands[0])
                if not prompt_path.is_file():
                    raise RuntimeError("encoder completed without creating prompt.pt")
                if not self.dry_run:
                    temporary = cached_prompt.with_suffix(f".pt.tmp.{os.getpid()}")
                    shutil.copyfile(prompt_path, temporary)
                    os.replace(temporary, cached_prompt)
                    self._append_log(task_id, f"[webui] prompt cache stored: {cached_prompt}")
            self._set_marker(task_id, "loading", 50, "encoding complete; loading inference model")
            if self.dry_run:
                self._run_dry_stage(task_id, commands[1], "inference")
            elif task.get("tensor_parallel"):
                self._ensure_tp_cache(task_id, task)
                self._run_tp_worker_task(task_id, task, run_dir)
            else:
                self._execute(task_id, commands[1])
            self._append_log(task_id, "[webui] inference process complete (denoise and decode)")
            output = run_dir / "output.mp4"
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("inference completed without creating output.mp4")
            self._append_log(task_id, f"[webui] done: {output}")
            self._update(task_id, status="done", stage="done", progress=100, error=None)
        except Exception as exc:  # subprocess failures must become visible task errors
            message = str(exc) or exc.__class__.__name__
            self._append_log(task_id, f"[webui] ERROR: {message}")
            self._update(task_id, status="error", stage="error", error=message)
        finally:
            with self._lock:
                if self._active_id == task_id:
                    self._active_id = None

    def close(self) -> None:
        """Stop the supervised process on server shutdown instead of leaving it orphaned."""
        with self._lock:
            self._closed = True
            process, worker = self._process, self._worker
        self._stop_tp_worker()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if worker is not None:
            worker.join(timeout=10)


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, list[str]], dict[str, list[UploadedFile]]]:
    header = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii", "strict")
        + body
    )
    message = BytesParser(policy=email_policy).parsebytes(header)
    if message.get_content_type() != "multipart/form-data" or not message.is_multipart():
        raise SubmissionError("POST /api/tasks requires multipart/form-data")
    fields: dict[str, list[str]] = {}
    uploads: dict[str, list[UploadedFile]] = {}
    for index, part in enumerate(message.iter_parts()):
        if index >= 40 or part.is_multipart():
            raise SubmissionError("too many or invalid multipart form fields")
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        name = "refs" if name == "refs[]" else name
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            # Browsers include filename="" for empty file inputs.
            if not filename and not payload:
                continue
            uploads.setdefault(name, []).append(
                UploadedFile(name, filename, payload, part.get_content_type())
            )
        else:
            charset = part.get_content_charset() or "utf-8"
            try:
                value = payload.decode(charset, errors="strict")
            except (LookupError, UnicodeError) as exc:
                raise SubmissionError("form fields must use valid text encoding") from exc
            fields.setdefault(name, []).append(value)
    return fields, uploads


def _read_static(name: str) -> bytes:
    if name not in {"index.html", "logs.html", "style.css", "app.js", "logs.js"}:
        raise FileNotFoundError(name)
    return (STATIC_ROOT / name).read_bytes()


class WebUIHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler_class, manager: TaskManager | None = None):
        self.task_manager = manager
        super().__init__(address, handler_class)

    def server_close(self) -> None:
        if self.task_manager is not None:
            self.task_manager.close()
        super().server_close()


class WebUIHandler(BaseHTTPRequestHandler):
    server: WebUIHTTPServer

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _video(self, path: Path) -> None:
        """Stream the MP4, with single byte ranges for browser playback and seeking."""
        with path.open("rb") as video:
            size = os.fstat(video.fileno()).st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                try:
                    if not match or not any(match.groups()):
                        raise ValueError
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                    else:
                        start = max(0, size - int(last))
                    if start > end or start >= size:
                        raise ValueError
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Disposition", 'inline; filename="output.mp4"')
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            video.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = video.read(min(128 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, status: int, payload: object) -> None:
        self._send(
            status,
            (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _not_found(self) -> None:
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    @staticmethod
    def _task_id(path: str) -> str | None:
        value = unquote(path)
        return value if TASK_ID_PATTERN.fullmatch(value) else None

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        manager = self.server.task_manager
        if path == "/":
            self._send(HTTPStatus.OK, _read_static("index.html"), "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path.removeprefix("/static/")
            try:
                body = _read_static(name)
            except (FileNotFoundError, OSError):
                self._not_found()
                return
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, body, f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
            return
        if path.startswith("/logs/"):
            task_id = self._task_id(path.removeprefix("/logs/"))
            if task_id is None or manager.get(task_id) is None:
                self._not_found()
                return
            page = _read_static("logs.html").decode("utf-8").replace(
                "{{TASK_ID}}", html.escape(task_id, quote=True)
            )
            self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/tasks":
            self._json(HTTPStatus.OK, {"tasks": manager.history()})
            return
        match = re.fullmatch(r"/api/tasks/([^/]+)(?:/(log|video))?", path)
        if not match:
            self._not_found()
            return
        task_id = self._task_id(match.group(1))
        if task_id is None or manager.get(task_id) is None:
            self._not_found()
            return
        suffix = match.group(2)
        if suffix == "log":
            try:
                log = manager.read_log(task_id)
            except (KeyError, OSError):
                self._not_found()
                return
            self._send(HTTPStatus.OK, log.encode("utf-8"), "text/plain; charset=utf-8")
        elif suffix == "video":
            video = manager.video_path(task_id)
            if video is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "video is not ready"})
                return
            self._video(video)
        else:
            self._json(HTTPStatus.OK, manager.get(task_id))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/tasks":
            self._not_found()
            return
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length is required"})
            return
        if length > MAX_TOTAL_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request is too large"})
            return
        content_type = self.headers.get("Content-Type", "")
        try:
            body = self.rfile.read(length)
            fields, uploads = _parse_multipart(content_type, body)
            submission = validate_submission(fields, uploads)
            task_id = self.server.task_manager.submit(submission)
        except TaskConflictError as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except SubmissionError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # keep an unexpected request failure as JSON
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._json(HTTPStatus.ACCEPTED, {"id": task_id})

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[webui] " + (format % args) + "\n")


def create_server(host: str = "0.0.0.0", port: int = 3000, runs_dir: Path | None = None) -> WebUIHTTPServer:
    # Bind first: a second launch on an occupied port must not mark the first
    # server's running tasks as interrupted while loading history.
    server = WebUIHTTPServer((host, port), WebUIHandler)
    try:
        server.task_manager = TaskManager(runs_dir or RUNS_DIR)
    except Exception:
        server.server_close()
        raise
    return server


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VDN-H3 browser inference UI")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "3000")))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    server = create_server(args.host, args.port)

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    print(f"VDN-H3 WebUI listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WebUI", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
