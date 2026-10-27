"""Delay-pattern utilities (MusicGen).

An alternative to the depth transformer: offset codebook k by k frames so that the
K codebooks at one output position become conditionally independent and can be
predicted in parallel by K heads on a plain transformer. Included for the repo
skeleton; the default model path (lm/model.py) uses the depth transformer instead.
"""
from __future__ import annotations

import torch


def apply_delay(codes: torch.Tensor, pad_id: int) -> torch.Tensor:
    """codes: (B, T, K) -> (B, T+K, K), codebook k delayed by k frames."""
    B, T, K = codes.shape
    out = torch.full((B, T + K, K), pad_id, dtype=codes.dtype, device=codes.device)
    for k in range(K):
        out[:, k:k + T, k] = codes[:, :, k]
    return out


def undo_delay(codes: torch.Tensor, T: int) -> torch.Tensor:
    """Inverse of apply_delay. codes: (B, T+K, K) -> (B, T, K)."""
    K = codes.shape[-1]
    return torch.stack([codes[:, k:k + T, k] for k in range(K)], dim=-1)
