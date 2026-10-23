"""Experimental two-NPU weight TP, with optional CPU block offload (torchrun)."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist

from src.config import load_config
from src.config.inference import InferenceConfig, validate_ablation, validate_kernels, validate_parallel
from src.inference.render import decode_and_save, generate_latents
from src.inference.utils import assemble
from src.inference.utils.assemble import build_inference_model, render_record, write_json
from src.inference.utils.base_shard import build_base_shard_model
from src.inference.utils.chunked_ff import install_chunked_feed_forward
from src.inference.utils.prompt_cache import load_prompt
from src.inference.utils.tensor_parallel import (TensorParallelRuntime, install_tensor_parallel,
                                                 prepare_tensor_parallel, storage_bytes,
                                                 storage_by_device,
                                                 validate_tp_launch)
from src.inference.utils.tp_cache import build_cached_model
from src.inference.utils.tp_cpu_assembly import cpu_assembly_threads, time_cpu_assembly
from src.inference.utils.tp_offload import apply_tp_offload
from src.inference.utils.tp_prefetch import enable_tp_prefetch
from src.utils.device import set_device, synchronize, empty_cache


def validate_tp_entrypoint(cfg):
    """Static entrypoint restrictions; torchrun ranks are checked only at runtime."""
    if not cfg.parallel.npu_tensor_parallel:
        raise ValueError("infer_tp.py requires parallel.npu_tensor_parallel=true")
    if cfg.render.device not in ("npu", "npu:0"):
        raise ValueError("infer_tp.py requires render.device=npu:0; local ranks select NPU0/1")


def main():
    started = time.perf_counter()
    cfg = load_config(InferenceConfig, extra_validators=[validate_ablation, validate_kernels,
                                                        validate_parallel, validate_tp_entrypoint])
    validate_tp_entrypoint(cfg)
    rank, local_rank = validate_tp_launch()
    device = torch.device(f"npu:{local_rank}")
    set_device(device)
    torch.npu.reset_peak_memory_stats(device)
    try:
        # Cold CPU assembly can take >15 minutes on one rank. Bound waits while
        # allowing rank-serialized loading and rank-zero decode to finish.
        dist.init_process_group("hccl", timeout=timedelta(minutes=45))
        if dist.get_world_size() != 2 or dist.get_rank() != rank:
            raise RuntimeError("TP process group does not match the validated local0/1 launch")
        runtime = TensorParallelRuntime(rank, 2, device, parity=cfg.parallel.npu_tp_parity)
        with torch.no_grad():
            _run(cfg, runtime, started)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def allocator_peaks(device):
    return {"allocated_bytes": torch.npu.max_memory_allocated(device),
            "reserved_bytes": torch.npu.max_memory_reserved(device)}


def _atomic_json(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _serve_loop(cfg, runtime, render_request, queue_dir: str) -> None:
    """Serve render requests from a file queue until a stop file arrives.

    Only rank zero reads the queue; each request is broadcast to rank one with
    ``dist.broadcast_object_list`` so both ranks enter the next forward together.
    """
    queue = os.path.abspath(queue_dir)
    os.makedirs(queue, exist_ok=True)
    request_path = os.path.join(queue, "request.json")
    pending_path = os.path.join(queue, "request.pending")
    result_path = os.path.join(queue, "result.json")
    stop_path = os.path.join(queue, "stop")
    ready_path = os.path.join(queue, "ready")
    if runtime.is_main:
        for path in (request_path, pending_path, result_path, stop_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        _atomic_json(ready_path, {"pid": os.getpid()})
        print("[tp-worker] ready", flush=True)
    try:
        last_sequence = None
        while True:
            request = None
            if runtime.is_main:
                while not os.path.exists(request_path) and not os.path.exists(stop_path):
                    time.sleep(0.1)
                if os.path.exists(request_path):
                    with open(request_path, "r", encoding="utf-8") as handle:
                        request = json.load(handle)
                    os.unlink(request_path)
                    request["_sequence"] = time.monotonic_ns()
                    _atomic_json(pending_path, request)
                else:
                    request = {"type": "stop"}
            else:
                while not os.path.exists(stop_path):
                    if os.path.exists(pending_path):
                        try:
                            with open(pending_path, "r", encoding="utf-8") as handle:
                                candidate = json.load(handle)
                        except (OSError, ValueError):
                            time.sleep(0.05)
                            continue
                        if candidate.get("_sequence") != last_sequence:
                            request = candidate
                            break
                    time.sleep(0.1)
                if request is None:
                    request = {"type": "stop"}
            box = [request]
            dist.broadcast_object_list(box, src=0)
            request = box[0]
            last_sequence = request.get("_sequence") if isinstance(request, dict) else None
            if not isinstance(request, dict):
                raise RuntimeError("TP worker received a non-object request")
            if request.get("type") == "stop":
                break
            try:
                result = render_request(request, keep_session=True)
                payload = {"status": "ok", **result}
            except BaseException as exc:
                payload = {"status": "error",
                           "error": f"{type(exc).__name__}: {exc}"}
                if runtime.is_main:
                    _atomic_json(result_path, payload)
                raise
            if runtime.is_main:
                _atomic_json(result_path, payload)
    finally:
        if runtime.is_main:
            try:
                os.unlink(ready_path)
            except FileNotFoundError:
                pass


def _run(cfg, runtime, started):
    device = runtime.device
    offload = cfg.render.offload_dit
    offload_manager = None
    install_chunked_feed_forward()

    def progress(stage, **fields):
        if runtime.is_main:
            print("[vdn-progress] " + json.dumps({"stage": stage, **fields}), flush=True)

    progress("loading")
    model = None
    if cfg.parallel.tp_cache:
        cache_dir = os.path.abspath(cfg.parallel.tp_cache)
        print(f"TP rank {runtime.rank}: loading cached merged shard from {cache_dir}",
              flush=True)
        model, shard_info, cache_load_seconds = build_cached_model(cfg, runtime, cache_dir)
        cpu_setup = {"cache_load_seconds": cache_load_seconds}
        print(f"TP rank {runtime.rank}: cache load {cache_load_seconds:.1f}s", flush=True)
        runtime.barrier()
    elif cfg.parallel.merge_device == "npu":
        # Both ranks read only their own shard; no full host model is assembled.
        print(f"TP rank {runtime.rank}: loading base shard and merging LoRAs on NPU",
              flush=True)
        model, shard_info, base_shard_seconds = build_base_shard_model(
            cfg, runtime, log=runtime.is_main)
        cpu_setup = {**shard_info, "base_shard_seconds": base_shard_seconds}
        print(f"TP rank {runtime.rank}: base shard {base_shard_seconds:.1f}s", flush=True)
        runtime.barrier()
    else:
        # Rank-serialized CPU assembly avoids two simultaneous full host models.
        for owner in range(runtime.world_size):
            if runtime.rank == owner:
                owner_started = time.perf_counter()
                cpu_setup = {}
                print(f"TP rank {owner}: assembling checkpoint and merging LoRAs on CPU",
                      flush=True)
                with cpu_assembly_threads(cpu_setup):
                    print(f"TP rank {owner}: CPU threads {cpu_setup['old_threads']} -> "
                          f"{cpu_setup['assembly_threads']} (assembly only)", flush=True)
                    build_started = time.perf_counter()
                    with time_cpu_assembly(assemble, cpu_setup):
                        model = build_inference_model(cfg, "cpu", install_offload=False,
                                                      load_decoders=runtime.is_main,
                                                      log=runtime.is_main,
                                                      softmax_device=runtime.device)
                    cpu_setup["build_seconds"] = time.perf_counter() - build_started
                    print(f"TP rank {owner}: slicing CPU weights before NPU transfer",
                          flush=True)
                    shard_started = time.perf_counter()
                    shard_info = install_tensor_parallel(model.transformer, runtime)
                    cpu_setup["shard_seconds"] = time.perf_counter() - shard_started
                    clear_started = time.perf_counter()
                    model.artifact.weights.clear()
                    gc.collect()
                    cpu_setup["clear_and_gc_seconds"] = time.perf_counter() - clear_started
                cpu_setup["cpu_seconds"] = time.perf_counter() - owner_started
                onload_started = time.perf_counter()
                if not offload or runtime.parity == "row_gather":
                    model.transformer.to(device)
                    synchronize(device)
                cpu_setup["onload_seconds"] = time.perf_counter() - onload_started
                cpu_setup["owner_setup_seconds"] = time.perf_counter() - owner_started
                print(f"TP rank {runtime.rank}: CPU setup {cpu_setup}", flush=True)
            runtime.barrier()
    # Parity buffers need both ranks' shards and HCCL collectives, so they cannot run
    # inside the rank-serialized assembly loop while the peer is waiting at a barrier.
    prepare_started = time.perf_counter()
    shard_info["parity_prepare"] = prepare_tensor_parallel(model.transformer)
    synchronize(device)
    shard_info["parity_prepare_seconds"] = time.perf_counter() - prepare_started
    if offload:
        # Hooks wrap the final TP forwards. They move only the current shard block;
        # registered nonpersistent row_gather buffers move with it as well.
        model.transformer.to("cpu")
        offload_strategy = cfg.parallel.tp_offload_strategy
        base_strategy = "retained" if offload_strategy == "prefetch" else offload_strategy
        offload_manager = apply_tp_offload(model.transformer, device, strategy=base_strategy)
        if offload_strategy == "prefetch":
            offload_manager = enable_tp_prefetch(offload_manager, model.transformer)
        gc.collect()
        empty_cache(device)
    shard_info["resident"] = storage_bytes(model.transformer)
    shard_info["storage_by_device_before_denoise"] = storage_by_device(model.transformer)
    shard_info["all_parameters_resident"] = all(
        p.device == device for p in model.transformer.parameters())
    if not offload and not shard_info["all_parameters_resident"]:
        raise RuntimeError("TP requires every transformer parameter resident on its NPU")
    if offload and any(t.device.type != "cpu" for t in
                       list(model.transformer.parameters()) + list(model.transformer.buffers())):
        raise RuntimeError("TP offload requires CPU-backed parameters and buffers between forwards")
    print(f"TP rank {runtime.rank}: resident={shard_info['resident']}, "
          f"all_parameters_resident={shard_info['all_parameters_resident']}, "
          f"cpu_block_offload={offload}", flush=True)
    setup_seconds = time.perf_counter() - started
    cpu_setup["setup_including_rank_wait_seconds"] = setup_seconds

    def render_request(request=None, keep_session=False):
        nonlocal offload_manager
        request_started = time.perf_counter()
        if request is not None:
            cfg.render.prompt_file = request["prompt_file"]
            cfg.render.out = request["out"]
            cfg.render.num_steps = int(request["num_steps"])
            cfg.render.num_frames = int(request["num_frames"])
            cfg.render.width = int(request["width"])
            cfg.render.height = int(request["height"])
            cfg.render.fps = int(request["fps"])
            cfg.render.seed = int(request["seed"])
            cfg.render.warmup_steps = int(request.get("warmup_steps", 0))
            cfg.render.save_latents = bool(request.get("save_latents", False))
            cfg.render.latents_root = request.get("latents_root") or cfg.render.latents_root
        prompt, tags, conditions = load_prompt(cfg.render.prompt_file, str(device))
        synchronize(device)

        def sample(steps, timings, callback=None):
            return generate_latents(model.transformer, prompt, tags, cfg.render.num_frames,
                                    steps, cfg.render.seed, device,
                                    video_shift=cfg.render.video_shift, audio_shift=cfg.render.audio_shift,
                                    runtime=runtime, step_seconds=timings, conditions=conditions,
                                    width=cfg.render.width, height=cfg.render.height,
                                    progress_callback=callback)

        if cfg.render.warmup_steps:
            progress("warmup")
            sample(cfg.render.warmup_steps, [])
        runtime.events.clear()
        runtime.collectives = 0
        if offload_manager is not None:
            offload_manager.reset_metrics()
        memory_phases = {"setup_including_warmup": allocator_peaks(device)}
        torch.npu.reset_peak_memory_stats(device)
        steps = []
        progress("denoise", completed=0, total=cfg.render.num_steps)
        denoise_started = time.perf_counter()
        video, audio = sample(cfg.render.num_steps, steps,
                              lambda step, total, elapsed: progress(
                                  "denoise", completed=step, total=total, seconds=round(elapsed, 3)))
        denoise_seconds = time.perf_counter() - denoise_started
        local = {**shard_info, **runtime.profile(), "cpu_setup": cpu_setup,
                 "parity": cfg.parallel.npu_tp_parity,
                 "storage_by_device_after_denoise": storage_by_device(model.transformer),
                 "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
                 "peak_reserved_bytes": torch.npu.max_memory_reserved(device)}
        if offload_manager is not None:
            local["offload_metrics"] = offload_manager.snapshot()
        memory_phases["denoise"] = allocator_peaks(device)
        torch.npu.reset_peak_memory_stats(device)
        release_started = time.perf_counter()
        # Offload hooks have already returned weights/buffers to CPU. A one-shot
        # render drops them before decode; a persistent worker keeps them for the
        # next request.
        if not keep_session:
            if not offload:
                model.transformer.to("cpu")
            else:
                close = getattr(offload_manager, "close_prefetch", None)
                if close is not None:
                    close()
                model.transformer = None
                offload_manager = None
        runtime.release_scratch()
        gc.collect()
        empty_cache(device)
        runtime.barrier()
        release_seconds = time.perf_counter() - release_started
        if runtime.is_main and cfg.render.save_latents:
            os.makedirs(cfg.render.latents_root, exist_ok=True)
            torch.save({"video": video.cpu(), "audio": audio.cpu()},
                       os.path.join(cfg.render.latents_root, os.path.basename(cfg.render.out) + ".latents.pt"))
        memory_phases["release_and_save_latents"] = allocator_peaks(device)
        torch.npu.reset_peak_memory_stats(device)
        decode_started = time.perf_counter()
        progress("decode")
        if runtime.is_main:
            decode_and_save(video, audio, model.vae, model.audio_vae, cfg.render.out, str(device), cfg.render.fps)
            synchronize(device)
        runtime.barrier()
        # A persistent worker outlives this request. Drop the decoded workspace
        # and return the VAE/decode allocator cache before the next request; a
        # following prompt encoder runs in a separate process on the same NPU.
        del video, audio
        gc.collect()
        empty_cache(device)
        runtime.barrier()
        decode_seconds = time.perf_counter() - decode_started
        memory_phases["decode_and_encode"] = allocator_peaks(device)
        local["allocator_peaks_by_phase"] = memory_phases
        local["end_to_end_allocator_peak"] = {
            key: max(phase[key] for phase in memory_phases.values())
            for key in ("allocated_bytes", "reserved_bytes")}
        local["allocator_measurement_coverage"] = "process setup through decode barrier; excludes final JSON metadata gathering"
        by_rank = [None] * runtime.world_size
        dist.all_gather_object(by_rank, local)
        if runtime.is_main:
            record = render_record(cfg, model)
            parallel_record = {"kind": "npu_tensor_parallel", "world_size": 2,
                               "backend": "hccl", "by_rank": by_rank,
                               "parity": cfg.parallel.npu_tp_parity,
                               "merge_device": cfg.parallel.merge_device,
                               "merge_delta_device": cfg.parallel.merge_delta_device,
                                "reduction_dtype": "float32", "cpu_offload": offload,
                                "offload_strategy": cfg.parallel.tp_offload_strategy if offload else "resident"}
            if cfg.parallel.tp_cache:
                parallel_record["tp_cache"] = os.path.abspath(cfg.parallel.tp_cache)
                parallel_record["tp_cache_load_seconds"] = [
                    (rank_record.get("cpu_setup") or {}).get("cache_load_seconds")
                    for rank_record in by_rank]
            if cfg.parallel.merge_device == "npu":
                parallel_record["base_shard"] = [
                    {"rank": index, **{
                        key: (rank_record.get("cpu_setup") or {}).get(key)
                        for key in ("base_read_seconds", "assign_seconds",
                                    "onload_seconds", "npu_merge_seconds")}}
                    for index, rank_record in enumerate(by_rank)]
            record["parallel"] = parallel_record
            record["timings"] = {"model_setup_seconds": 0.0 if request else setup_seconds,
                                 "session_setup_seconds": setup_seconds,
                                 "denoise_seconds": denoise_seconds, "step_seconds": steps,
                                 "release_seconds": release_seconds,
                                 "decode_and_encode_seconds": decode_seconds,
                                 "end_to_end_seconds": time.perf_counter() - request_started}
            write_json(record, cfg.render.out + ".inference.json")
            print(record["timings"], flush=True)
            return {"timings": record["timings"], "out": cfg.render.out}
        return {"timings": None, "out": cfg.render.out}

    queue_dir = os.environ.get("VDN_TP_WORKER_QUEUE")
    if queue_dir:
        _serve_loop(cfg, runtime, render_request, queue_dir)
    else:
        render_request(keep_session=False)


if __name__ == "__main__":
    main()
