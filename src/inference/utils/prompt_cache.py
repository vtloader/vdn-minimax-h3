"""The prompt cache: the .pt that encode_prompt.py and encode_keyframes.py write and every
render reads. What its anchors mean is said once, here, so the writers and the reader
agree by import.

    prompt, prompt_embeds (L, 5120) bf16, text_token_tags (L,)        every cache
    keyframe_anchors, condition_latents [(1, 24, 1, h, w) fp32 ...]    keyframes or references
"""
from dataclasses import dataclass

import torch

# The anchor encode_keyframes.py gives a reference image (`--refs`); a cache whose anchors
# are all of it is a ref2va-like request and takes the ref2va layout in generate_latents.
REFERENCE_ANCHOR = "ref"
KEYFRAME_MODES = {("first",): "i2va", ("last",): "l2va", ("first", "last"): "fl2va"}


def is_reference_request(anchors):
    return bool(anchors) and all(anchor == REFERENCE_ANCHOR for anchor in anchors)


def conditioning_mode(anchors):
    """What a cache's anchors make the request: i2va, l2va or fl2va by which keyframes are
    anchored, ref2va when every anchor is a reference."""
    if is_reference_request(anchors):
        return "ref2va"
    return KEYFRAME_MODES[tuple(anchors)]


@dataclass(frozen=True)
class ReferenceConditions:
    """The conditioning rows of a keyframe or ref2va prompt cache."""
    anchors: tuple
    condition_latents: list
    audio_condition_latents: list
    kinds: tuple
    has_audio: tuple


def load_prompt(prompt_file: str, device: str):
    """A prompt cache from encode_prompt.py (t2va) or encode_keyframes.py (keyframes or
    references); both carry prompt_embeds and text_token_tags. Returns (prompt_embeds,
    text_token_tags, conditions); `conditions` is a ReferenceConditions for a keyframe
    or reference cache, else None."""
    text = torch.load(prompt_file, map_location="cpu", weights_only=True)
    conditions = None
    if text.get("keyframe_anchors"):
        anchors = tuple(text["keyframe_anchors"])
        kinds = tuple(text.get("reference_kinds") or ("image",) * len(anchors))
        has_audio = tuple(text.get("reference_has_audio") or (False,) * len(anchors))
        conditions = ReferenceConditions(
            anchors=anchors,
            condition_latents=[c.to(device, torch.float32) for c in text["condition_latents"]],
            audio_condition_latents=[c.to(device, torch.float32)
                                     for c in text.get("audio_condition_latents", [])],
            kinds=kinds,
            has_audio=has_audio,
        )
    return text["prompt_embeds"].to(device, torch.bfloat16), text["text_token_tags"], conditions
