"""Codec losses: multi-resolution mel reconstruction + LSGAN adversarial + feature matching.

LSGAN is used to match the HiFi-GAN discriminators (least-squares targets):
  D loss   = mean((D(real) - 1)^2) + mean(D(fake)^2)
  G adv    = mean((D(fake) - 1)^2)
  feat mat = L1 between D's intermediate features for real vs fake
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class MultiResMelLoss(nn.Module):
    """L1 between log-mel spectrograms at several STFT window sizes.

    Several window sizes together force both sharp timing (small n_fft) and sharp
    frequency (large n_fft), which is what makes the audio intelligible.
    """

    def __init__(self, sr: int = 16000, n_ffts=(512, 1024, 2048), n_mels: int = 64):
        super().__init__()
        self.mels = nn.ModuleList([
            MelSpectrogram(sample_rate=sr, n_fft=n, hop_length=n // 4,
                           win_length=n, n_mels=n_mels, power=1.0)
            for n in n_ffts
        ])

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # inputs (B, 1, L); MelSpectrogram accepts (..., L)
        real, fake = real.squeeze(1), fake.squeeze(1)
        loss = real.new_zeros(())
        for mel in self.mels:
            r = mel(real).clamp(min=1e-5).log()
            f = mel(fake).clamp(min=1e-5).log()
            loss = loss + F.l1_loss(r, f)
        return loss / len(self.mels)


def discriminator_loss(real_scores, fake_scores) -> torch.Tensor:
    loss = 0.0
    for r, f in zip(real_scores, fake_scores):
        loss = loss + (r - 1).pow(2).mean() + f.pow(2).mean()
    return loss / len(real_scores)


def generator_adv_loss(fake_scores) -> torch.Tensor:
    loss = 0.0
    for f in fake_scores:
        loss = loss + (f - 1).pow(2).mean()
    return loss / len(fake_scores)


def feature_matching_loss(real_feats, fake_feats) -> torch.Tensor:
    loss = 0.0
    n = 0
    for rf, ff in zip(real_feats, fake_feats):          # per sub-discriminator
        for r, f in zip(rf, ff):                        # per layer
            loss = loss + F.l1_loss(r.detach(), f)
            n += 1
    return loss / max(n, 1)
