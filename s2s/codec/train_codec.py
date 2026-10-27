"""Train the neural codec (encoder + RVQ + decoder) with a HiFi-GAN-style GAN.

    python -m s2s.codec.train_codec --config configs/codec.yaml
    python -m s2s.codec.train_codec --config configs/codec.yaml batch_size=8 crop_seconds=0.6

Resume automatically picks up the latest checkpoint in cfg.ckpt_dir.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from ..audio_io import save_audio
from ..data.audio_dataset import AudioCropDataset
from ..utils import (AverageMeter, amp_dtype, build_optimizer, count_params,
                     get_device, load_checkpoint, load_config, save_checkpoint,
                     set_seed)
from .discriminators import Discriminator
from .losses import (MultiResMelLoss, discriminator_loss, feature_matching_loss,
                     generator_adv_loss)
from .model import NeuralCodec


def latest_ckpt(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not cks:
        return None
    return os.path.join(ckpt_dir, max(cks, key=lambda f: int(f.split("_")[-1][:-3])))


def main():
    cfg = load_config("configs/codec.yaml")
    set_seed(cfg.seed)
    device = get_device()
    dtype = amp_dtype(cfg.precision)
    use_amp = device.type == "cuda" and cfg.precision != "fp32"

    audio_root = os.path.join(cfg.data_root, "LibriSpeech", cfg.subset)
    ds = AudioCropDataset(audio_root, cfg.sample_rate, cfg.crop_seconds)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=cfg.num_workers > 0)
    print(f"{len(ds)} files | frame-crop {cfg.crop_seconds}s | batch {cfg.batch_size}")

    codec = NeuralCodec(cfg.dim, cfg.base, tuple(cfg.strides),
                        cfg.codebook_size, cfg.n_q, cfg.sample_rate).to(device)
    disc = Discriminator().to(device)
    mel_loss = MultiResMelLoss(cfg.sample_rate, tuple(cfg.mel_n_ffts), cfg.n_mels).to(device)
    print(f"codec {count_params(codec)/1e6:.1f}M params | frame rate {codec.frame_rate:.1f} Hz "
          f"| disc {count_params(disc)/1e6:.1f}M")

    opt_g = build_optimizer(codec.parameters(), cfg.lr, cfg.get("optimizer", "adamw"))
    opt_d = build_optimizer(disc.parameters(), cfg.get("disc_lr", cfg.lr), cfg.get("optimizer", "adamw"))

    step = 0
    ckpt = latest_ckpt(cfg.ckpt_dir)
    if ckpt:
        blob = load_checkpoint(ckpt, map_location=device, codec=codec, disc=disc,
                               opt_g=opt_g, opt_d=opt_d)
        step = blob.get("step", 0)
        print(f"resumed from {ckpt} at step {step}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(cfg.ckpt_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    meters = {k: AverageMeter() for k in ["mel", "adv", "fm", "commit", "d", "active"]}
    codec.train(); disc.train()
    t0 = time.time()
    data_iter = iter(dl)

    while step < cfg.max_steps:
        try:
            wav = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            wav = next(data_iter)
        wav = wav.to(device, non_blocking=True)                   # (B, 1, L)

        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            recon, codes, commit = codec(wav)

        # ---- discriminator step ----
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            real_s, _ = disc(wav)
            fake_s, _ = disc(recon.detach())
            d_loss = discriminator_loss(real_s, fake_s)
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        # ---- generator step ----
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            real_s, real_f = disc(wav)
            fake_s, fake_f = disc(recon)
            adv = generator_adv_loss(fake_s)
            fm = feature_matching_loss(real_f, fake_f)
            mel = mel_loss(wav, recon)
            g_loss = (cfg.lambda_mel * mel + cfg.lambda_adv * adv
                      + cfg.lambda_fm * fm + cfg.lambda_commit * commit)
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        # ---- logging ----
        active = codes.reshape(-1, cfg.n_q)[:, 0].unique().numel()  # codebook-0 usage
        meters["mel"].update(mel.item()); meters["adv"].update(adv.item())
        meters["fm"].update(fm.item()); meters["commit"].update(commit.item())
        meters["d"].update(d_loss.item()); meters["active"].update(active)
        step += 1

        if step % cfg.log_every == 0:
            sps = cfg.log_every / (time.time() - t0); t0 = time.time()
            print(f"step {step:>7} | mel {meters['mel'].avg:.3f} | adv {meters['adv'].avg:.3f} "
                  f"| fm {meters['fm'].avg:.3f} | commit {meters['commit'].avg:.4f} "
                  f"| d {meters['d'].avg:.3f} | cb0 {meters['active'].avg:.0f}/{cfg.codebook_size} "
                  f"| {sps:.1f} it/s")
            for m in meters.values():
                m.reset()

        if step % cfg.sample_every == 0:
            with torch.no_grad():
                save_audio(os.path.join(sample_dir, f"orig_{step}.wav"),
                           wav[0].cpu(), cfg.sample_rate)
                save_audio(os.path.join(sample_dir, f"recon_{step}.wav"),
                           recon[0].float().clamp(-1, 1).cpu(), cfg.sample_rate)

        if step % cfg.ckpt_every == 0:
            path = os.path.join(cfg.ckpt_dir, f"codec_{step}.pt")
            save_checkpoint(path, step, codec=codec, disc=disc, opt_g=opt_g,
                            opt_d=opt_d, config=dict(cfg))
            print("saved", path)

    print("training complete")


if __name__ == "__main__":
    main()
