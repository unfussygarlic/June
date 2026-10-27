"""Residual Vector Quantization, DAC-style (L2-normalized, factorized codes).

Each quantizer projects the latent down to a small `code_dim`, L2-normalizes it, and
quantizes on the unit sphere. This decouples codebook lookup from the model dimension
and makes quantization scale-invariant, which is the main fix for codebook collapse:
utilization typically goes from ~25% to 80%+ versus plain euclidean VQ.

Updates use EMA (online k-means) + dead-code restart. Straight-through estimator passes
gradients to the encoder. Commitment loss lives in the normalized code space, so it stays
O(1) instead of exploding as the encoder's latent scale grows.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, code_dim: int = 8,
                 decay: float = 0.95, eps: float = 1e-5, restart_threshold: float = 1.0):
        super().__init__()
        self.K = codebook_size
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold

        self.in_proj = nn.Linear(dim, code_dim)       # dim -> code_dim
        self.out_proj = nn.Linear(code_dim, dim)       # code_dim -> dim

        embed = l2norm(torch.randn(codebook_size, code_dim))
        self.register_buffer("embed", embed)                       # (K, code_dim) unit norm
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor):
        """x: (B, T, dim) -> (out (B,T,dim), codes (B,T), commit scalar)."""
        z = l2norm(self.in_proj(x))                                # (B, T, code_dim)
        flat = z.reshape(-1, self.code_dim)
        sim = flat @ self.embed.t()                                # (N, K) cosine (unit norm)
        idx = sim.argmax(1)
        onehot = F.one_hot(idx, self.K).type_as(flat)
        q = (onehot @ self.embed).view_as(z)                       # (B, T, code_dim)

        if self.training:
            self._ema_update(flat, onehot)

        commit = F.mse_loss(z, q.detach())                         # O(1): both unit norm
        q = z + (q - z).detach()                                   # straight-through
        out = self.out_proj(q)                                     # (B, T, dim)
        return out, idx.view(x.shape[:-1]), commit

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        flat = flat.float()
        onehot = onehot.float()
        n = onehot.sum(0)
        embed_sum = flat.t() @ onehot                              # (code_dim, K)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

        total = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
        self.embed.copy_(l2norm(self.embed_avg / cs.unsqueeze(1)))  # keep codes unit norm

        dead = self.cluster_size < self.restart_threshold
        n_dead = int(dead.sum())
        if n_dead > 0:
            pick = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
            self.embed[dead] = flat[pick]
            self.embed_avg[dead] = flat[pick]
            self.cluster_size[dead] = 1.0

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T) int -> (B, T, dim)."""
        q = F.embedding(codes, self.embed)                         # (B, T, code_dim)
        return self.out_proj(q)


class ResidualVQ(nn.Module):
    def __init__(self, dim: int, codebook_size: int, n_q: int, code_dim: int = 8,
                 **vq_kwargs):
        super().__init__()
        self.n_q = n_q
        self.layers = nn.ModuleList(
            [VectorQuantizer(dim, codebook_size, code_dim, **vq_kwargs) for _ in range(n_q)]
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, dim) -> (q_out (B,T,dim), codes (B,T,n_q), commit scalar)."""
        residual = x
        q_out = torch.zeros_like(x)
        codes, commit = [], 0.0
        for vq in self.layers:
            out, idx, c = vq(residual)
            residual = residual - out
            q_out = q_out + out
            codes.append(idx)
            commit = commit + c
        return q_out, torch.stack(codes, dim=-1), commit

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T, n_q) -> (B, T, dim)."""
        out = 0.0
        for i, vq in enumerate(self.layers):
            out = out + vq.decode(codes[..., i])
        return out
