"""Generate a speech continuation from a seed clip. Self-configures the LM from the
checkpoint, so it works for any trained architecture (d=512/768/1024, ...).

    python -m s2s.lm.sample --config configs/lm.yaml \
        lm_ckpt=checkpoints/lm/lm_best.pt \
        seed_wav=data/LibriSpeech/dev-clean/<...>.flac \
        gen_seconds=4 temperature=1.0 top_k=250 top_p=0.95 \
        repetition_penalty=1.2 rep_window=50 out=out.wav

Sampling knobs (all overridable on the CLI):
  temperature        higher = more varied; low temps cause silence/droning
  top_k / top_p      truncate the distribution (top_p is nucleus sampling)
  repetition_penalty >1 discourages repeating recent tokens (fights silence collapse)
  rep_window         how many recent frames the penalty looks back over
"""
from __future__ import annotations

import torch

from ..audio_io import load_audio, save_audio
from ..codec.model import load_codec
from ..utils import get_device, load_config
from .model import load_speechlm


def main():
    cfg = load_config("configs/lm.yaml")
    device = get_device()

    codec = load_codec(cfg.codec_ckpt, device)
    model, mcfg = load_speechlm(cfg.lm_ckpt, device)
    n_q = mcfg.get("n_q", cfg.n_q)
    sr = codec.sample_rate
    max_ctx = model.pos.size(1)

    gen_frames = int(cfg.get("gen_seconds", 4) * codec.frame_rate)
    temperature = cfg.get("temperature", 0.9)
    top_k = cfg.get("top_k", 250)
    top_p = cfg.get("top_p", 0.0)
    rep = cfg.get("repetition_penalty", 1.0)
    rep_w = cfg.get("rep_window", 50)

    # seed -> tokens (1, T0, n_q)
    wav, _ = load_audio(cfg.seed_wav, sr)
    wav = wav / wav.abs().max().clamp(min=1e-5) * 0.95
    codes = codec.encode(wav.unsqueeze(0).to(device))

    for _ in range(gen_frames):
        ctx = codes[:, -max_ctx:]
        h_t = model.temporal_ctx(ctx)[:, -1]                  # (1, d)
        recent = codes[:, -rep_w:].transpose(1, 2) if rep > 1 else None  # (1, n_q, M)
        frame = model.depth_sample(h_t, temperature, top_k, top_p, rep, recent)
        codes = torch.cat([codes, frame.unsqueeze(1)], dim=1)

    out_wav = codec.decode(codes).float().clamp(-1, 1).cpu()[0]
    out_path = cfg.get("out", "continuation.wav")
    save_audio(out_path, out_wav, sr)
    gen_s = gen_frames / codec.frame_rate
    print(f"wrote {out_path} ({out_wav.shape[-1]/sr:.1f}s total, last {gen_s:.1f}s generated) "
          f"| temp {temperature} top_k {top_k} top_p {top_p} rep {rep}")


if __name__ == "__main__":
    main()