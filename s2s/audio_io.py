"""Audio I/O via soundfile (libsndfile) — robust across torchaudio versions, which
now route load/save through torchcodec. Handles wav/flac/ogg. Returns mono tensors.
"""
from __future__ import annotations

import soundfile as sf
import torch


def load_audio(path: str, target_sr: int | None = None):
    """-> (wav (1, L) float tensor, sr). Downmixed to mono, optionally resampled."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)   # (L, C)
    wav = torch.from_numpy(data.T).mean(0, keepdim=True)        # mono (1, L)
    if target_sr and sr != target_sr:
        import torchaudio.functional as AF
        wav = AF.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_audio(path: str, wav: torch.Tensor, sr: int) -> None:
    """wav: (1, L) or (L,) tensor -> write a mono file."""
    if wav.dim() == 2:
        wav = wav[0]
    sf.write(path, wav.detach().cpu().numpy(), sr)
