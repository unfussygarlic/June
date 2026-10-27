"""Residual Vector Quantization.

VectorQuantizer: nearest-codebook lookup with an EMA (k-means) codebook update,
a straight-through estimator so gradients reach the encoder, a commitment loss,
and dead-code restart (the #1 fix for codebook collapse).

ResidualVQ: a stack of VectorQuantizers, each quantizing the residual of the last.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99,
                 eps: float = 1e-5, restart_threshold: float = 1.0):
        super().__init__()
        self.dim = dim
        self.K = codebook_size
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold

        embed = torch.randn(codebook_size, dim)
        self.register_buffer("embed", embed)                       # (K, D) the codebook
        self.register_buffer("cluster_size", torch.zeros(codebook_size))  # (K,)
        self.register_buffer("embed_avg", embed.clone())           # (K, D)

    def forward(self, x: torch.Tensor):
        """x: (B, T, D) -> (quantized (B,T,D), codes (B,T), commit_loss scalar)."""
        flat = x.reshape(-1, self.dim)                             # (N, D), N = B*T
        # squared euclidean distance to every code: ||x||^2 - 2 x.e + ||e||^2
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))                        # (N, K)
        idx = dist.argmin(1)                                       # (N,) nearest code
        onehot = F.one_hot(idx, self.K).type_as(flat)             # (N, K)
        q = (onehot @ self.embed).view_as(x)                      # gather -> (B, T, D)

        if self.training:
            self._ema_update(flat, onehot)

        commit = F.mse_loss(x, q.detach())                        # train encoder toward code
        q = x + (q - x).detach()                                  # straight-through
        return q, idx.view(x.shape[:-1]), commit

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        flat = flat.float()          # buffers are fp32; autocast feeds bf16 -> cast back
        onehot = onehot.float()
        n = onehot.sum(0)                                        # (K,) count this batch
        embed_sum = flat.t() @ onehot                            # (D, K) sum per code
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

        total = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
        self.embed.copy_(self.embed_avg / cs.unsqueeze(1))       # centroid = sum / count

        # restart dead codes with random live encoder outputs (prevents collapse)
        dead = self.cluster_size < self.restart_threshold
        n_dead = int(dead.sum())
        if n_dead > 0:
            pick = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
            self.embed[dead] = flat[pick]
            self.embed_avg[dead] = flat[pick]
            self.cluster_size[dead] = 1.0

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T) int -> (B, T, D) the chosen code vectors."""
        return F.embedding(codes, self.embed)


class ResidualVQ(nn.Module):
    def __init__(self, dim: int, codebook_size: int, n_q: int, **vq_kwargs):
        super().__init__()
        self.n_q = n_q
        self.layers = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size, **vq_kwargs) for _ in range(n_q)]
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, D) -> (q_out (B,T,D), codes (B,T,n_q), commit scalar)."""
        residual = x
        q_out = torch.zeros_like(x)
        codes, commit = [], 0.0
        for vq in self.layers:
            q, idx, c = vq(residual)
            residual = residual - q
            q_out = q_out + q
            codes.append(idx)
            commit = commit + c
        return q_out, torch.stack(codes, dim=-1), commit          # codes: (B, T, n_q)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T, n_q) -> (B, T, D) summed reconstruction."""
        out = 0.0
        for i, vq in enumerate(self.layers):
            out = out + vq.decode(codes[..., i])
        return out
