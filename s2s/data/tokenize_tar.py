"""Tokenize audio straight out of a .tar.gz, WITHOUT extracting it to disk.

    python -m s2s.data.tokenize_tar --config configs/lm.yaml \
        archive=data/train-clean-360.tar.gz subset=train-clean-360

For when you don't have room to extract the whole corpus. It streams the archive
once, reads each FLAC from memory, encodes it with the codec, and writes only the
small token .npy to token_dir (same layout as tokenize.py). The audio is never
written to disk, so you only ever need the .tar.gz itself. Delete the archive when
it finishes.

Resumable: re-running skips members whose .npy already exists (it still has to stream
past them, since a gzip tar can't be randomly seeked, but it won't re-encode).
"""
from __future__ import annotations

import os
import tarfile

import numpy as np
import torch

from ..audio_io import load_audio_fileobj
from ..codec.model import load_codec
from ..utils import get_device, load_config

EXTS = (".flac", ".wav", ".ogg")


def rel_key(name: str, subset: str) -> str:
    """LibriSpeech/train-clean-360/spk/ch/utt.flac -> spk/ch/utt (token relpath)."""
    marker = f"{subset}/"
    i = name.find(marker)
    rel = name[i + len(marker):] if i >= 0 else os.path.basename(name)
    return os.path.splitext(rel)[0]


def main():
    cfg = load_config("configs/lm.yaml")
    device = get_device()

    archive = cfg.get("archive")
    assert archive and os.path.exists(archive), \
        "pass archive=path/to/subset.tar.gz"
    subset = cfg.get("subset", "train-clean-360")
    token_dir = cfg.token_dir
    os.makedirs(token_dir, exist_ok=True)

    codec = load_codec(cfg.codec_ckpt, device)
    sr = codec.sample_rate
    print(f"streaming {archive} -> {token_dir} (subset {subset}, {sr} Hz)")

    done, skipped, failed = 0, 0, 0
    with tarfile.open(archive, "r|gz") as tar:          # pure streaming, one pass
        for m in tar:
            if not m.isfile() or not m.name.lower().endswith(EXTS):
                continue
            out = os.path.join(token_dir, rel_key(m.name, subset) + ".npy")
            if os.path.exists(out):
                skipped += 1
                continue
            f = tar.extractfile(m)                       # read before advancing
            if f is None:
                continue
            try:
                wav, _ = load_audio_fileobj(f.read(), sr)
                wav = wav / wav.abs().max().clamp(min=1e-5) * 0.95
                codes = codec.encode(wav.unsqueeze(0).to(device))[0]      # (T, n_q)
                codes = codes.transpose(0, 1).cpu().numpy().astype(np.uint16)  # (n_q, T)
            except Exception as e:
                print(f"  skip {m.name}: {e}")
                failed += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, codes)
            done += 1
            if done % 500 == 0:
                print(f"  tokenized {done} (skipped {skipped}, failed {failed})")

    print(f"done: {done} new, {skipped} already present, {failed} failed")
    print(f"you can now delete {archive} to free space:  rm {archive}")


if __name__ == "__main__":
    main()