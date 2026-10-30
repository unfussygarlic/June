"""Dataset of precomputed codec tokens for LM training.

Each .npy file holds codes of shape (channels, T) as uint16, where
  channels = n_q           for single-stream (continuation), or
  channels = 2 * n_q       for two-stream dialogue (user codebooks then agent).
Returns a random window (channels, window_frames), PAD-padded if the clip is short.
"""
from __future__ import annotations

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def list_token_files(token_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(token_dir, "**", "*.npy"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"no token shards under {token_dir!r}. Run: "
            f"python -m s2s.data.tokenize --config configs/lm.yaml"
        )
    return files


class TokenWindowDataset(Dataset):
    def __init__(self, token_dir: str, window_frames: int, pad_id: int,
                 files: list[str] | None = None):
        # `files` lets the caller pass an explicit train/val split; else glob the dir.
        self.files = files if files is not None else list_token_files(token_dir)
        self.W = window_frames
        self.pad = pad_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        codes = np.load(self.files[i])                 # (C, T) uint16
        C, T = codes.shape
        if T >= self.W:
            s = random.randint(0, T - self.W)
            win = codes[:, s:s + self.W]
        else:
            win = np.full((C, self.W), self.pad, dtype=codes.dtype)
            win[:, :T] = codes
        return torch.from_numpy(win.astype(np.int64))  # (C, W)