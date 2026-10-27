"""HiFi-GAN discriminators: Multi-Period (MPD) + Multi-Scale (MSD).

These are the standard discriminators reused by EnCodec and DAC. Each returns both
a scalar-ish "realness" map and a list of intermediate feature maps (used for the
feature-matching loss). We expose a single `Discriminator` that runs both and
concatenates their outputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class DiscriminatorP(nn.Module):
    """Reshapes the 1-D waveform into 2-D with a given period, then 2-D convs."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)           # (B,1,T/p,p)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class DiscriminatorS(nn.Module):
    """1-D convolutions directly on the (possibly downsampled) waveform."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class Discriminator(nn.Module):
    """Runs MPD (periods 2,3,5,7,11) + MSD (3 scales) and returns aggregated lists."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.mpd = nn.ModuleList([DiscriminatorP(p) for p in periods])
        self.msd = nn.ModuleList([DiscriminatorS() for _ in range(3)])
        self.pools = nn.ModuleList([nn.Identity(),
                                    nn.AvgPool1d(4, 2, padding=2),
                                    nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y: torch.Tensor):
        """y: (B, 1, L) -> (list of score tensors, list of feature-map lists)."""
        scores, feats = [], []
        for d in self.mpd:
            s, f = d(y)
            scores.append(s)
            feats.append(f)
        x = y
        for pool, d in zip(self.pools, self.msd):
            x = pool(x)
            s, f = d(x)
            scores.append(s)
            feats.append(f)
        return scores, feats
