"""Tiny forward/backward checks on CPU — no data, no downloads. Catches shape and
gradient bugs before you commit GPU hours.

    python -m s2s.smoke_test
"""
from __future__ import annotations

import torch

from .codec.discriminators import Discriminator
from .codec.losses import (MultiResMelLoss, discriminator_loss,
                           feature_matching_loss, generator_adv_loss)
from .codec.model import NeuralCodec
from .codec.quantizer import ResidualVQ
from .lm.delay import apply_delay, undo_delay
from .lm.model import SpeechLM, loss_fn


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_rvq():
    print("ResidualVQ")
    rvq = ResidualVQ(dim=16, codebook_size=32, n_q=4).train()
    x = torch.randn(2, 5, 16)
    q, codes, commit = rvq(x)
    check("quantized shape", q.shape == x.shape)
    check("codes shape (B,T,n_q)", codes.shape == (2, 5, 4))
    check("codes in range", int(codes.max()) < 32 and int(codes.min()) >= 0)
    check("commit is scalar", commit.dim() == 0)
    check("decode roundtrips shape", rvq.decode(codes).shape == x.shape)


def test_codec():
    print("NeuralCodec + discriminators + losses (full GAN step)")
    sr = 16000
    codec = NeuralCodec(dim=16, base=8, strides=(2, 4, 5, 8),
                        codebook_size=32, n_q=4, sample_rate=sr).train()
    disc = Discriminator().train()
    mel = MultiResMelLoss(sr=sr, n_ffts=(256, 512), n_mels=32)
    L = codec.total_stride * 8
    wav = torch.randn(2, 1, L) * 0.1
    recon, codes, commit = codec(wav)
    check("recon length matches input", recon.shape == wav.shape)
    check("frame count = L / stride", codes.shape[1] == L // codec.total_stride)

    # discriminator step
    real_s, _ = disc(wav)
    fake_s, _ = disc(recon.detach())
    d_loss = discriminator_loss(real_s, fake_s)
    d_loss.backward()
    check("disc loss finite + backward", torch.isfinite(d_loss))

    # generator step
    real_s, real_f = disc(wav)
    fake_s, fake_f = disc(recon)
    g = 15 * mel(wav, recon) + generator_adv_loss(fake_s) \
        + 2 * feature_matching_loss(real_f, fake_f) + 0.25 * commit
    g.backward()
    enc_grad = next(codec.encoder.parameters()).grad
    check("gen loss finite", torch.isfinite(g))
    check("gradient reaches encoder (straight-through works)",
          enc_grad is not None and enc_grad.abs().sum() > 0)


def test_lm(n_streams):
    print(f"SpeechLM (n_streams={n_streams})")
    n_q, card, d = 4, 32, 32
    model = SpeechLM(n_streams=n_streams, n_q=n_q, codebook_size=card, d=d,
                     temporal_layers=2, depth_layers=2, heads=2, max_frames=64).train()
    B, T = 2, 12
    C = n_streams * n_q
    codes_in = torch.randint(0, card, (B, T, C))
    tgt = torch.randint(0, card, (B, T, n_q))
    logits = model(codes_in, tgt)
    check("logits shape (B,T,n_q,card)", logits.shape == (B, T, n_q, card))
    loss = loss_fn(logits, tgt, pad=card)
    loss.backward()
    check("loss finite + backward", torch.isfinite(loss))
    frame = model.depth_sample(torch.randn(B, d))
    check("depth_sample shape (B,n_q)", frame.shape == (B, n_q))
    check("sampled codes in range", int(frame.max()) < card)


def test_delay():
    print("delay pattern")
    codes = torch.randint(0, 32, (2, 10, 4))
    delayed = apply_delay(codes, pad_id=32)
    back = undo_delay(delayed, T=10)
    check("delay roundtrip", torch.equal(back, codes))


if __name__ == "__main__":
    torch.manual_seed(0)
    test_rvq()
    test_codec()
    test_lm(1)
    test_lm(2)
    test_delay()
    print("\nAll smoke tests passed.")
