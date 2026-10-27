"""Generate a speech continuation: encode a seed clip, let the LM keep talking, decode.

    python -m s2s.lm.sample --config configs/lm.yaml \
        seed_wav=path/to/prompt.flac lm_ckpt=checkpoints/lm/lm_200000.pt \
        gen_seconds=5 out=continuation.wav

Single-stream (continuation). Temporal context is recomputed each frame (no KV cache)
for simplicity — fine for a few seconds; add a cache for real-time use.
"""
from __future__ import annotations

import torch

from ..audio_io import load_audio, save_audio
from ..codec.model import load_codec
from ..utils import get_device, load_checkpoint, load_config
from .model import SpeechLM


def main():
    cfg = load_config("configs/lm.yaml")
    device = get_device()

    codec = load_codec(cfg.codec_ckpt, device)

    model = SpeechLM(cfg.n_streams, cfg.n_q, cfg.codebook_size, cfg.d,
                     cfg.temporal_layers, cfg.depth_layers, cfg.heads,
                     max_frames=cfg.window_frames + 8).to(device)
    load_checkpoint(cfg.lm_ckpt, map_location=device, model=model)
    model.eval()

    sr = codec.sample_rate
    max_ctx = model.pos.size(1)
    gen_frames = int(cfg.get("gen_seconds", 5) * codec.frame_rate)
    temperature = cfg.get("temperature", 0.8)
    top_k = cfg.get("top_k", 200)

    # seed codes (1, T0, n_q)
    wav, _ = load_audio(cfg.seed_wav, sr)
    wav = wav / wav.abs().max().clamp(min=1e-5) * 0.95
    codes = codec.encode(wav.unsqueeze(0).to(device))             # (1, T0, n_q)

    for _ in range(gen_frames):
        ctx = codes[:, -max_ctx:]
        h_t = model.temporal_ctx(ctx)[:, -1]                      # (1, d)
        frame = model.depth_sample(h_t, temperature, top_k)       # (1, n_q)
        codes = torch.cat([codes, frame.unsqueeze(1)], dim=1)

    out_wav = codec.decode(codes).float().clamp(-1, 1).cpu()[0]
    out_path = cfg.get("out", "continuation.wav")
    save_audio(out_path, out_wav, sr)
    print(f"wrote {out_path} ({out_wav.shape[-1] / sr:.1f}s)")


if __name__ == "__main__":
    main()
