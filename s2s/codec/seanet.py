"""SEANet-style convolutional encoder and decoder.

Encoder: Conv1d stack with strided downsampling. Product of `strides` is the total
downsampling factor, so frame_rate = sample_rate / prod(strides).
  strides=(2,4,5,8) -> 320x  -> 16000/320 = 50 Hz
  strides=(2,4,5,8,4) -> 1280x -> 12.5 Hz (Mimi-like)

Decoder mirrors it with ConvTranspose1d. The final length is reconciled to the
input length by the NeuralCodec wrapper (see model.py), so odd strides are fine.
"""
from __future__ import annotations

from math import prod

import torch
import torch.nn as nn


class ResUnit(nn.Module):
    """Residual block that keeps length and channels fixed; dilation widens context."""

    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation, padding=dilation),
            nn.ELU(),
            nn.Conv1d(ch, ch, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, dim: int = 128, base: int = 32, strides=(2, 4, 5, 8),
                 n_residual: int = 2):
        super().__init__()
        self.strides = tuple(strides)
        self.total_stride = prod(self.strides)

        layers = [nn.Conv1d(1, base, kernel_size=7, padding=3)]
        ch = base
        for s in strides:
            for d in range(n_residual):
                layers.append(ResUnit(ch, dilation=3 ** d))
            layers += [
                nn.ELU(),
                nn.Conv1d(ch, ch * 2, kernel_size=2 * s, stride=s, padding=(s + 1) // 2),
            ]
            ch *= 2
        layers += [nn.ELU(), nn.Conv1d(ch, dim, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.out_channels = dim

    def forward(self, x):
        """x: (B, 1, L) -> (B, dim, T)."""
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, dim: int = 128, base: int = 32, strides=(2, 4, 5, 8),
                 n_residual: int = 2):
        super().__init__()
        rev = tuple(reversed(strides))
        ch = base * (2 ** len(strides))
        layers = [nn.Conv1d(dim, ch, kernel_size=3, padding=1)]
        for s in rev:
            layers += [
                nn.ELU(),
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=2 * s, stride=s,
                                   padding=(s + 1) // 2, output_padding=s % 2),
            ]
            ch //= 2
            for d in range(n_residual):
                layers.append(ResUnit(ch, dilation=3 ** d))
        layers += [nn.ELU(), nn.Conv1d(ch, 1, kernel_size=7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """z: (B, dim, T) -> (B, 1, ~L)."""
        return self.net(z)
