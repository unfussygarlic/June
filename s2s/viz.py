"""Spectrogram comparison PNGs for easy visual inspection (upload them for review).

Gracefully no-ops if matplotlib isn't installed.
"""
from __future__ import annotations

import torch


def save_spectrogram_compare(orig: torch.Tensor, recon: torch.Tensor, path: str,
                             sr: int = 16000, n_fft: int = 1024, n_mels: int = 128) -> bool:
    """orig/recon: (1, L) or (L,) tensors. Writes a stacked orig/recon mel-spec PNG.
    Returns True if written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from torchaudio.transforms import MelSpectrogram
    except Exception:
        return False

    o = orig.detach().float().cpu().flatten()
    r = recon.detach().float().cpu().flatten()
    mel = MelSpectrogram(sr, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels, power=2.0)

    def db(x):
        return (mel(x).clamp(min=1e-5).log10() * 10).numpy()

    So, Sr = db(o), db(r)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, S, t in zip(ax, [So, Sr], ["ORIGINAL", "RECON"]):
        a.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-50, vmax=20)
        a.set_title(t)
        a.set_ylabel("mel bin")
    ax[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return True
