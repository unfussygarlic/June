"""Encode a whole audio corpus to codec tokens, saved as .npy shards (run once).

    python -m s2s.data.tokenize --config configs/lm.yaml
    # add more data later (same codec, same token_dir -> bigger training set):
    python -m s2s.data.tokenize --config configs/lm.yaml subset=train-clean-360

Reads the trained codec at cfg.codec_ckpt, walks every audio file under the chosen
subset, and writes token_dir/<relpath>.npy of shape (n_q, T) uint16. Different
LibriSpeech subsets have disjoint speaker folders, so tokenizing several into the
same token_dir just grows the dataset with no collisions.
This is the single most important step for fitting LM training on 8 GB: afterwards
the codec and raw audio never touch VRAM again.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from ..audio_io import load_audio
from ..codec.model import load_codec
from ..utils import get_device, load_config


def main():
    cfg = load_config("configs/lm.yaml")
    codec_cfg = load_config_codec()
    device = get_device()

    codec = load_codec(cfg.codec_ckpt, device)     # architecture comes from the checkpoint
    sr = codec.sample_rate

    # subset override lets you tokenize extra corpora into the same token_dir
    subset = cfg.get("subset", codec_cfg["subset"])
    audio_root = cfg.get("audio_root", os.path.join(
        codec_cfg["data_root"], "LibriSpeech", subset))
    files = []
    for ext in (".flac", ".wav", ".mp3", ".ogg"):
        files += glob.glob(os.path.join(audio_root, "**", f"*{ext}"), recursive=True)
    files.sort()
    print(f"tokenizing {len(files)} files from {subset} -> {cfg.token_dir}")

    os.makedirs(cfg.token_dir, exist_ok=True)
    for path in tqdm(files):
        try:
            wav, _ = load_audio(path, sr)
            wav = wav / wav.abs().max().clamp(min=1e-5) * 0.95
            codes = codec.encode(wav.unsqueeze(0).to(device))[0]   # (T, n_q)
            codes = codes.transpose(0, 1).cpu().numpy().astype(np.uint16)  # (n_q, T)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rel = os.path.splitext(os.path.relpath(path, audio_root))[0]
        out = os.path.join(cfg.token_dir, rel + ".npy")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, codes)
    print("done")


def load_config_codec() -> dict:
    """Read configs/codec.yaml directly (no CLI parsing)."""
    import yaml
    with open("configs/codec.yaml") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()