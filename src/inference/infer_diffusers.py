"""Render through the PUBLISHED diffusers components, not this repository's stack.

    python src/inference/infer_diffusers.py                       # prompts/example_0.pt
    python src/inference/infer_diffusers.py "a prompt" --out results/diffusers.mp4
    python src/inference/infer_diffusers.py "a prompt" --steps 50 \
        --transformer stage-b-step-2000/diffusers
    python src/inference/infer_diffusers.py "a prompt" \
        --first prompts/image/first.png --last prompts/image/last.png
    python src/inference/infer_diffusers.py --offload_dit          # a 24 GB card

This runs the README's `Load it with Diffusers` snippet. Everything it renders comes from
the Hub, so the patched diffusers that scripts/setup_diffusers.sh installs is the whole
setup; the only thing it reads from here is the default prompt, and a prompt of your own
replaces that. It deliberately shares NO code with infer.py and infer_ulysses.py -- those
are the fast stack (fp8, the decomposed window kernel, Ulysses); this is the portable
one, single-GPU, bf16 or torchao fp8.

`workflow=` keeps the unused 61.7 GB transformer partition from being fetched. Every
model but the transformer is offloaded, always: the 62 GB Qwen3-VL text encoder comes
onto the GPU one layer at a time, and each decoder whole, while it runs. The transformer
stays on the GPU unless `--offload_dit` streams it in one block at a time too.
"""
import argparse
import os

import torch
from accelerate import cpu_offload_with_hook
from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import encode_video

from src.utils.device import device_of, generator

REPO = "OpenVDN/vdn-minimax-h3"
FPS = 24
# The repository's own showcase prompt. Every prompt cache carries the text it was
# encoded from, so the default is that text rather than a second copy of it here.
DEFAULT_PROMPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "prompts", "example_0.pt")


def offload(pipe, device, dit=False):
    """The text encoder streams in by layer. The decoders come in whole through
    accelerate's hook: the pipeline calls their `encode` and `decode`, and that is the
    only hook those fire. With `dit` the transformer streams in by block, and otherwise
    it stays on the GPU. It offloads whole or by block, never by leaf: its fused kernels
    read a child's weights without calling the child, so a leaf's hook never fires."""
    use_stream = device_of(device).type == "cuda"
    apply_group_offloading(pipe.text_encoder, onload_device=device, offload_device="cpu",
                           offload_type="leaf_level", use_stream=use_stream)
    _, vae = cpu_offload_with_hook(pipe.vae, execution_device=device)
    cpu_offload_with_hook(pipe.audio_vae, execution_device=device, prev_module_hook=vae)

    def decoder_back(module, args):
        vae.offload()                 # fl2va encodes its keyframes before denoising

    pipe.transformer.register_forward_pre_hook(decoder_back)
    if not dit:
        pipe.transformer.to(device)
        return
    apply_group_offloading(pipe.transformer, onload_device=device, offload_device="cpu",
                           offload_type="block_level", num_blocks_per_group=1,
                           use_stream=use_stream)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompt", nargs="?",
                   help=f"defaults to the text {os.path.basename(DEFAULT_PROMPT)} was "
                        "encoded from")
    p.add_argument("--out", default="results/diffusers.mp4")
    p.add_argument("--steps", type=int, default=8,
                   help="model evaluations (NFE). The scheduler counts sigma grid "
                        "points, one more than that, and this passes the +1 for you")
    p.add_argument("--frames", type=int, default=345,
                   help="snapped up to the next 17n+5; 5 to 15 seconds at 24 fps. The "
                        "default is the 14.4 seconds the reported numbers use, which "
                        "one 140 GB GPU holds in bf16 with room to spare")
    p.add_argument("--transformer", default=None,
                   help="a checkpoint's diffusers/ subfolder. Default: whatever the "
                        "repository's index names, the 8-step model")
    p.add_argument("--first", help="keyframe the video starts from")
    p.add_argument("--last", help="keyframe the video ends on")
    p.add_argument("--fp8", action="store_true",
                   help="the component's preset TorchAoConfig (pip install torchao): every "
                        "wide Linear in fp8 e4m3, the transformer's weights drop from 62 GB "
                        "to 45. A TorchAoConfig of your own goes through quantization_config= "
                        "in load_components; any other backend raises")
    p.add_argument("--offload_dit", action="store_true",
                   help="stream the transformer onto the GPU one block at a time, which "
                        "a 24 GB card needs: 345 frames then peak at 22 GB, 20 in fp8. The "
                        "transformer offloads whole or by block, never by leaf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default=REPO, help="Hub repository or local weights root")
    args = p.parse_args()
    if args.fp8 and device_of(args.device).type != "cuda":
        p.error("--fp8 is only supported by the CUDA TorchAO path")
    prompt = args.prompt or torch.load(DEFAULT_PROMPT, map_location="cpu",
                                       weights_only=True)["prompt"]

    keyframes = {}
    if args.first or args.last:
        from diffusers.utils import load_image

        if args.first:
            keyframes["image"] = load_image(args.first)
        if args.last:
            keyframes["last_image"] = load_image(args.last)

    pipe = ModularPipeline.from_pretrained(args.model, workflow="fl2va" if keyframes else "t2va")

    if device_of(args.device).type == "npu":
        # Published remote code predates this repository's NPU dispatch. Load the
        # component with the local, adapted class before loading the other components.
        from src.diffusers_export.modeling import VDNMiniMaxH3Transformer3DModel

        transformer = VDNMiniMaxH3Transformer3DModel.from_pretrained(
            args.model,
            subfolder=args.transformer or "stage-dmd-step-250/diffusers",
            torch_dtype=torch.bfloat16,
            softmax_backend="ref",
        )
        pipe.update_components(transformer=transformer)

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if args.transformer:
        load_kwargs["subfolder"] = {"transformer": args.transformer}
    if args.fp8:
        # A dict keys a kwarg to one component; the text encoder would not know it.
        load_kwargs["fp8"] = {"transformer": True}
    pipe.load_components(**load_kwargs)
    offload(pipe, torch.device(args.device), dit=args.offload_dit)

    videos, audio, rate = pipe(
        prompt=prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps + 1,
        generator=generator(args.device, args.seed),
        output=["videos", "audio", "sampling_rate"],
        **keyframes,
    ).values()

    import numpy as np

    frames = torch.from_numpy(np.stack([np.asarray(frame) for frame in videos[0]]))
    encode_video(frames, fps=FPS, output_path=args.out,
                 audio=audio[0].float().cpu(), audio_sample_rate=rate)
    print(f"wrote {args.out}: {len(videos[0])} frames, "
          f"{audio.shape[-1] / rate:.2f}s of audio")


if __name__ == "__main__":
    main()
