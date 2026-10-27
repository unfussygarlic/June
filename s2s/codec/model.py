"""NeuralCodec = Encoder -> ResidualVQ -> Decoder.

This is your tokenizer (ears) and vocoder (voice) in one trainable module.
  forward(wav) -> (recon, codes, commit)   for training
  encode(wav)  -> codes (B, T, n_q)         for building token datasets
  decode(codes)-> wav (B, 1, L)             for generation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import ResidualVQ
from .seanet import Decoder, Encoder


class NeuralCodec(nn.Module):
    def __init__(self, dim=128, base=32, strides=(2, 4, 5, 8),
                 codebook_size=1024, n_q=8, sample_rate=16000, code_dim=8):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_q = n_q
        self.encoder = Encoder(dim, base, strides)
        self.decoder = Decoder(dim, base, strides)
        self.rvq = ResidualVQ(dim, codebook_size, n_q, code_dim=code_dim)
        self.total_stride = self.encoder.total_stride

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.total_stride

    def _match_len(self, recon: torch.Tensor, length: int) -> torch.Tensor:
        if recon.shape[-1] > length:
            return recon[..., :length]
        if recon.shape[-1] < length:
            return F.pad(recon, (0, length - recon.shape[-1]))
        return recon

    def forward(self, wav: torch.Tensor):
        """wav: (B, 1, L) -> recon (B,1,L), codes (B,T,n_q), commit scalar."""
        z = self.encoder(wav).transpose(1, 2)              # (B, T, dim)
        zq, codes, commit = self.rvq(z)                    # (B, T, dim)
        recon = self.decoder(zq.transpose(1, 2))           # (B, 1, ~L)
        return self._match_len(recon, wav.shape[-1]), codes, commit

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        z = self.encoder(wav).transpose(1, 2)
        _, codes, _ = self.rvq(z)
        return codes                                       # (B, T, n_q)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, length: int | None = None) -> torch.Tensor:
        zq = self.rvq.decode(codes)                        # (B, T, dim)
        recon = self.decoder(zq.transpose(1, 2))           # (B, 1, ~L)
        return recon if length is None else self._match_len(recon, length)


def load_codec(ckpt_path: str, device="cpu"):
    """Rebuild a NeuralCodec from a checkpoint, using the ARCHITECTURE stored in it
    (so it always matches the trained weights, regardless of current YAML)."""
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = blob.get("config", {})
    codec = NeuralCodec(
        dim=c.get("dim", 128), base=c.get("base", 32),
        strides=tuple(c.get("strides", (2, 4, 5, 8))),
        codebook_size=c.get("codebook_size", 1024), n_q=c.get("n_q", 8),
        sample_rate=c.get("sample_rate", 16000), code_dim=c.get("code_dim", 8),
    )
    codec.load_state_dict(blob["codec"])
    return codec.to(device).eval()
