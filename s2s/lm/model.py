"""Textless speech LM: a temporal transformer over frames + a depth transformer
over the K codebooks within each frame (the AudioLM / Moshi "RQ-Transformer").

Vocabulary = codec tokens. No text. Index `codebook_size` is reserved as PAD/BOS
(embeddings have codebook_size + 1 rows; heads predict codebook_size classes and
training ignores PAD targets).

n_streams: 1 = continuation (one speaker), 2 = full-duplex dialogue (user + agent).
In the 2-stream case, codes_in packs both streams' codebooks on the channel axis
(user 0..K-1 then agent 0..K-1) and the model predicts the agent's K codebooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Stack(nn.Module):
    """Causal transformer encoder stack (pre-norm, GELU).

    When grad_checkpointing=True, each layer is recomputed in the backward pass
    instead of storing its activations — trades ~30% compute for a large drop in
    activation memory. This is what lets a big model (e.g. d=1024) fit on 8 GB.
    """

    def __init__(self, d: int, n_layers: int, n_heads: int, grad_checkpointing: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d, n_heads, dim_feedforward=4 * d, batch_first=True,
                activation="gelu", norm_first=True)
            for _ in range(n_layers)
        ])
        self.grad_checkpointing = grad_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, src_mask=mask)
        return x


def sample_logits(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 200):
    """logits: (B, card) -> sampled token (B,)."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


class SpeechLM(nn.Module):
    def __init__(self, n_streams=1, n_q=8, codebook_size=1024, d=512,
                 temporal_layers=8, depth_layers=3, heads=8, max_frames=4096,
                 grad_checkpointing=False):
        super().__init__()
        self.n_streams, self.n_q, self.card, self.d = n_streams, n_q, codebook_size, d
        self.pad = codebook_size                                  # reserved index

        self.embeds = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, d) for _ in range(n_streams * n_q)])
        self.pos = nn.Parameter(torch.zeros(1, max_frames, d))
        self.temporal = Stack(d, temporal_layers, heads, grad_checkpointing)

        self.depth_in = nn.Embedding(codebook_size + 1, d)
        self.depth_pos = nn.Parameter(torch.zeros(1, n_q, d))
        self.depth = Stack(d, depth_layers, heads, grad_checkpointing)
        self.out = nn.ModuleList([nn.Linear(d, codebook_size) for _ in range(n_q)])

    def frame_embed(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, T, n_streams*n_q) -> (B, T, d) (summed codebook embeddings)."""
        e = self.embeds[0](codes[:, :, 0])
        for i in range(1, len(self.embeds)):
            e = e + self.embeds[i](codes[:, :, i])
        return e

    def temporal_ctx(self, codes_in: torch.Tensor) -> torch.Tensor:
        T = codes_in.size(1)
        return self.temporal(self.frame_embed(codes_in) + self.pos[:, :T])

    def forward(self, codes_in: torch.Tensor, agent_tgt: torch.Tensor) -> torch.Tensor:
        """codes_in: (B, T, n_streams*n_q), agent_tgt: (B, T, n_q)
        -> logits (B, T, n_q, card)."""
        B, T, _ = agent_tgt.shape
        h = self.temporal_ctx(codes_in)                           # (B, T, d)

        dep = self.depth_in(agent_tgt)                            # (B, T, K, d)
        dep = torch.cat([h.unsqueeze(2), dep[:, :, :-1]], dim=2)  # prepend context
        dep = (dep + self.depth_pos.unsqueeze(0)).reshape(B * T, self.n_q, self.d)
        dep = self.depth(dep)                                     # (B*T, K, d)
        logits = torch.stack([self.out[k](dep[:, k]) for k in range(self.n_q)], dim=1)
        return logits.view(B, T, self.n_q, self.card)

    @torch.no_grad()
    def depth_sample(self, h_t: torch.Tensor, temperature=0.8, top_k=200) -> torch.Tensor:
        """Sample one frame's K agent codebooks given context h_t: (B, d) -> (B, n_q)."""
        seq = h_t.unsqueeze(1)                                    # (B, 1, d)
        codes = []
        for k in range(self.n_q):
            x = seq + self.depth_pos[:, :seq.size(1)]
            hk = self.depth(x)[:, -1]                             # (B, d)
            tok = sample_logits(self.out[k](hk), temperature, top_k)   # (B,)
            codes.append(tok)
            seq = torch.cat([seq, self.depth_in(tok).unsqueeze(1)], dim=1)
        return torch.stack(codes, dim=-1)                        # (B, n_q)


def loss_fn(logits: torch.Tensor, target: torch.Tensor, pad: int) -> torch.Tensor:
    """Cross-entropy over (B, T, n_q); PAD targets are ignored."""
    B, T, K, C = logits.shape
    return F.cross_entropy(logits.reshape(-1, C), target.reshape(-1),
                           ignore_index=pad)