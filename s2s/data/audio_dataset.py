"""Raw-audio dataset for codec training: random fixed-length crops, resampled to mono."""
from __future__ import annotations

import glob
import os
import random

import torch
from torch.utils.data import Dataset

from ..audio_io import load_audio


class AudioCropDataset(Dataset):
    def __init__(self, root: str, sample_rate: int = 16000, crop_seconds: float = 0.8,
                 extensions=(".flac", ".wav", ".ogg")):
        self.sr = sample_rate
        self.crop_len = int(sample_rate * crop_seconds)
        self.files = []
        for ext in extensions:
            self.files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
        self.files.sort()
        if not self.files:
            raise FileNotFoundError(
                f"no audio under {root!r}. Run: python -m s2s.data.download --config configs/codec.yaml"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        # retry a few neighbours if a file is unreadable
        for j in range(4):
            try:
                wav, _ = load_audio(self.files[(i + j) % len(self.files)], self.sr)
                break
            except Exception:
                continue
        else:
            wav = torch.zeros(1, self.crop_len)

        L = wav.shape[-1]
        if L >= self.crop_len:
            start = random.randint(0, L - self.crop_len)
            wav = wav[:, start:start + self.crop_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.crop_len - L))
        # peak-normalize for stable training
        peak = wav.abs().max().clamp(min=1e-5)
        return wav / peak * 0.95                    # (1, crop_len)
