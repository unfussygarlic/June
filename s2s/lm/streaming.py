"""Streaming inference with a KV cache for the temporal transformer.

The offline sampler recomputes the whole temporal context every frame (O(T) per
step -> O(T^2) for a clip). This caches each temporal layer's keys/values so a new
frame costs O(T) attention against the cache instead, giving real-time (or faster)
generation. It reimplements the TransformerEncoderLayer(norm_first) math by hand so
it reuses the *trained* weights exactly — verified to match the non-cached path.

The depth transformer is left as-is: its sequence length is just n_q (<=8), so
caching it saves nothing.

Context is capped at the model's positional length (max_frames); streaming past that
needs a sliding-window cache (a later addition).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import sample_logits


class TemporalKVCache:
    def __init__(self, n_layers: int):
        self.k = [None] * n_layers        # per layer: (B, T, d)
        self.v = [None] * n_layers
        self.t = 0                        # frames in the cache (= next position index)

    def reset(self):
        for i in range(len(self.k)):
            self.k[i] = self.v[i] = None
        self.t = 0


def _layer_step(layer, x: torch.Tensor, cache: TemporalKVCache, i: int) -> torch.Tensor:
    """One TransformerEncoderLayer(norm_first=True) applied to the new frame x (B,1,d),
    attending to cached keys/values. Mirrors nn.TransformerEncoderLayer exactly (eval,
    no dropout)."""
    attn = layer.self_attn
    nh, d = attn.num_heads, x.size(-1)
    hd = d // nh

    xn = F.layer_norm(x, (d,), layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
    qkv = F.linear(xn, attn.in_proj_weight, attn.in_proj_bias)      # (B,1,3d)
    q, k, v = qkv.chunk(3, dim=-1)                                  # each (B,1,d)

    cache.k[i] = k if cache.k[i] is None else torch.cat([cache.k[i], k], dim=1)
    cache.v[i] = v if cache.v[i] is None else torch.cat([cache.v[i], v], dim=1)
    K, V = cache.k[i], cache.v[i]                                   # (B,T,d)

    def heads(t):
        return t.view(t.size(0), t.size(1), nh, hd).transpose(1, 2)  # (B,nh,T,hd)

    # new frame attends to all cached frames -> no mask needed (all are causal-valid)
    o = F.scaled_dot_product_attention(heads(q), heads(K), heads(V))  # (B,nh,1,hd)
    o = o.transpose(1, 2).reshape(x.size(0), 1, d)
    o = F.linear(o, attn.out_proj.weight, attn.out_proj.bias)
    x = x + o

    xn2 = F.layer_norm(x, (d,), layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
    x = x + layer.linear2(F.gelu(layer.linear1(xn2)))
    return x


class StreamingSpeechLM:
    """Wraps a trained SpeechLM for incremental, cached generation."""

    def __init__(self, model):
        self.m = model.eval()
        self.cache = TemporalKVCache(len(model.temporal.layers))

    def reset(self):
        self.cache.reset()

    @torch.no_grad()
    def step(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """Feed one input frame's codes (B, n_streams*n_q), advance the cache, and
        return the temporal context vector h (B, d) for predicting the next frame."""
        e = self.m.embeds[0](frame_codes[:, 0])
        for j in range(1, len(self.m.embeds)):
            e = e + self.m.embeds[j](frame_codes[:, j])
        x = (e + self.m.pos[:, self.cache.t]).unsqueeze(1)          # (B,1,d) + abs pos
        for i, layer in enumerate(self.m.temporal.layers):
            x = _layer_step(layer, x, self.cache, i)
        self.cache.t += 1
        return x[:, -1]                                             # (B, d)


@torch.no_grad()
def generate_stream(model, prompt_codes, gen_frames, temperature=1.0, top_k=250,
                    top_p=0.95, repetition_penalty=1.3, rep_window=50, on_frame=None):
    """prompt_codes: (1, P, n_q) -> full codes (1, P+gen_frames, n_q), using the KV cache.
    Single-stream. `on_frame(frame)` is called per generated frame (for live playback)."""
    stream = StreamingSpeechLM(model)
    max_ctx = model.pos.size(1)
    # absolute positions cap total context at max_ctx; trim the prompt so prompt+gen fit
    keep = max(1, min(prompt_codes.size(1), max_ctx - gen_frames))
    codes = prompt_codes[:, -keep:]
    # prime the cache with the prompt; the last step's h predicts the first new frame
    h = None
    for t in range(codes.size(1)):
        h = stream.step(codes[:, t])
    for _ in range(gen_frames):
        recent = codes[:, -rep_window:].transpose(1, 2) if repetition_penalty > 1 else None
        frame = model.depth_sample(h, temperature, top_k, top_p, repetition_penalty, recent)
        codes = torch.cat([codes, frame.unsqueeze(1)], dim=1)
        if on_frame is not None:
            on_frame(frame)
        h = stream.step(frame)
    return codes