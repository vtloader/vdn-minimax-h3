"""Encode t2va prompts with the Qwen3-VL conditioner and cache the results.

Replicates `MiniMaxH3TextEncoderStep` exactly: the prompt verbatim, no chat template, no
special tokens, hidden state after decoder layer 50, tags all text. Run once on one GPU/NPU;
inference then reuses the cached file and never loads the 62 GB conditioner again.

    python src/inference/encode_prompt.py --prompt "..." --out prompts/my_prompt.pt
    python src/inference/encode_prompt.py --jsonl prompts.jsonl --take 4 \
        --out_dir prompts --name prompt
    python src/inference/encode_prompt.py --prompt "..." --out prompts/my_prompt.pt \
        --device npu:1 --offload

The batch form exists because loading the conditioner costs far more than encoding: four
separate --prompt runs would pay the 62 GB load four times. It also keeps a prompt set
reproducible -- the file it reads is the dataset as published, not text pasted onto a
command line.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from src.paths import upstream_snapshot
from src.utils.device import device_of, set_device

TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def encode(processor, text_encoder, prompt, out, device):
    token_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    print(f"Encoding {len(token_ids)} text tokens on {device} -> {out}", flush=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device
    )
    with torch.no_grad():
        outputs = text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16).cpu()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,  # (L, 5120) bf16
        "text_token_tags": torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long),
    }, out)
    print(f"wrote {out}: {len(token_ids)} tokens, embeds {tuple(prompt_embeds.shape)}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description="Cache Qwen3-VL text conditioning on CUDA or NPU.")
    p.add_argument("--prompt", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--jsonl", type=str,
                   help="one JSON object per line with a 'prompt' field; batch mode")
    p.add_argument("--take", type=int, default=4, help="jsonl: encode the first N rows")
    p.add_argument("--rows", type=int, nargs="*", default=None,
                   help="jsonl: encode exactly these row indices instead of the first "
                        "--take; output names keep the ORIGINAL index "
                        "(<name>_<row>.pt), so a prompt's identity is stable no matter "
                        "which subset a batch encodes")
    p.add_argument("--out_dir", type=str, default="prompts")
    p.add_argument("--name", type=str, default="prompt",
                   help="jsonl: outputs are <out_dir>/<name>_<i>.pt")
    p.add_argument("--model_root", type=str, default=None,
                   help="a MiniMax-H3 snapshot with processor/ and text_encoder/; "
                        "default: downloaded from the Hub on first use")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="encoding device, e.g. cuda:0 or npu:1")
    p.add_argument("--offload", action="store_true",
                   help="keep encoder weights on CPU and move leaf modules to --device "
                        "only while encoding (lower accelerator memory, slower; no stream prefetch)")
    args = p.parse_args()
    if not args.jsonl and not (args.prompt and args.out):
        p.error("pass --jsonl, or both --prompt and --out")

    device = device_of(args.device)
    set_device(device)
    if args.model_root is None:
        print("Resolving processor and text encoder snapshot", flush=True)
        args.model_root = upstream_snapshot("processor", "text_encoder")

    print(f"Loading processor from {args.model_root}", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(args.model_root, subfolder="processor")
    print(f"Loading text encoder ({'CPU leaf offload' if args.offload else device})", flush=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_root, subfolder="text_encoder", dtype=torch.bfloat16
    )
    text_encoder.eval().requires_grad_(False)
    if args.offload:
        from diffusers.hooks import apply_group_offloading

        # Encoding calls the inner model directly, bypassing the outer LM forward.
        apply_group_offloading(
            text_encoder.model, onload_device=device, offload_device="cpu",
            offload_type="leaf_level", use_stream=False,
        )
    else:
        text_encoder.to(device)

    if args.jsonl:
        with open(args.jsonl) as f:
            all_rows = [json.loads(l) for l in f]
        picked = (list(enumerate(all_rows))[:args.take] if args.rows is None
                  else [(i, all_rows[i]) for i in args.rows])
        for i, r in picked:
            encode(processor, text_encoder, r["prompt"],
                   os.path.join(args.out_dir, f"{args.name}_{i}.pt"), device)
    else:
        encode(processor, text_encoder, args.prompt, args.out, device)


if __name__ == "__main__":
    main()
