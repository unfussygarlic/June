"""Streaming continuation with the KV cache, and a real-time-factor report.

    python -m s2s.lm.sample_stream --config configs/lm.yaml \
        lm_ckpt=checkpoints/lm/lm_best.pt \
        seed_wav=data/LibriSpeech/dev-clean/<...>.flac \
        gen_seconds=4 out=stream.wav

Prints the real-time factor (RTF): generated audio seconds / wall-clock seconds.
RTF > 1 means faster than real time. Same sampling knobs as sample.py.
"""
from __future__ import annotations

import time

import torch

from ..audio_io import load_audio, save_audio
from ..codec.model import load_codec
from ..utils import get_device, load_config
from .model import load_speechlm
from .streaming import generate_stream


def main():
    cfg = load_config("configs/lm.yaml")
    device = get_device()

    codec = load_codec(cfg.codec_ckpt, device)
    model, _ = load_speechlm(cfg.lm_ckpt, device)
    sr = codec.sample_rate
    gen_frames = int(cfg.get("gen_seconds", 4) * codec.frame_rate)

    wav, _ = load_audio(cfg.seed_wav, sr)
    wav = wav / wav.abs().max().clamp(min=1e-5) * 0.95
    prompt = codec.encode(wav.unsqueeze(0).to(device))          # (1, P, n_q)

    t0 = time.time()
    codes = generate_stream(
        model, prompt, gen_frames,
        temperature=cfg.get("temperature", 1.0), top_k=cfg.get("top_k", 250),
        top_p=cfg.get("top_p", 0.95), repetition_penalty=cfg.get("repetition_penalty", 1.3),
        rep_window=cfg.get("rep_window", 50))
    gen_wall = time.time() - t0

    out_wav = codec.decode(codes).float().clamp(-1, 1).cpu()[0]
    out_path = cfg.get("out", "stream.wav")
    save_audio(out_path, out_wav, sr)

    gen_s = gen_frames / codec.frame_rate
    rtf = gen_s / max(gen_wall, 1e-6)
    print(f"wrote {out_path} | generated {gen_s:.1f}s in {gen_wall:.2f}s wall "
          f"-> RTF {rtf:.2f}x ({'faster' if rtf>1 else 'slower'} than real time) "
          f"| {gen_frames/max(gen_wall,1e-6):.0f} frames/s")


if __name__ == "__main__":
    main()