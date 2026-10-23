"""Sequence-chunked SwiGLU evaluation for long packed sequences.

The packed ref2va layout carries the full reference video alongside the target
video.  At 1920x1088/243f that roughly doubles the token count, and the SwiGLU
projection's 2*ffn activation is several GiB per rank even with row_stream TP.
Chunking only the activation's token axis bounds that temporary without changing
the arithmetic.  The following output projection still runs once per block, so
this does not multiply the TP all-gather count.
"""
from __future__ import annotations

import os

import torch

from diffusers.models.activations import SwiGLU


DEFAULT_CHUNK_TOKENS = 65536


def _chunk_tokens() -> int:
    try:
        return int(os.environ.get("VDN_FF_CHUNK_TOKENS", str(DEFAULT_CHUNK_TOKENS)))
    except ValueError:
        return DEFAULT_CHUNK_TOKENS


def install_chunked_feed_forward() -> None:
    """Wrap SwiGLU.forward once, before any hooks capture bound forwards."""
    if getattr(SwiGLU, "_vdn_chunked_forward", False):
        return
    original = SwiGLU.forward

    def forward(self, hidden_states):
        chunk = _chunk_tokens()
        sequence_dim = 1 if hidden_states.ndim >= 3 else 0
        length = hidden_states.shape[sequence_dim]
        if chunk <= 0 or torch.is_grad_enabled() or length <= chunk:
            return original(self, hidden_states)
        outputs = []
        for start in range(0, length, chunk):
            index = [slice(None)] * hidden_states.ndim
            index[sequence_dim] = slice(start, min(start + chunk, length))
            outputs.append(original(self, hidden_states[tuple(index)]))
        return torch.cat(outputs, dim=sequence_dim)

    forward.__name__ = original.__name__
    forward.__doc__ = original.__doc__
    SwiGLU.forward = forward
    SwiGLU._vdn_chunked_forward = True
