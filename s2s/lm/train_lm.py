"""Train the textless speech LM on precomputed codec tokens.

    python -m s2s.lm.train_lm --config configs/lm.yaml
    python -m s2s.lm.train_lm --config configs/lm.yaml n_streams=2   # dialogue

Next-frame prediction: given frames up to t, predict frame t+1's K codebooks.

At each checkpoint it also:
  - computes a held-out VALIDATION loss and saves lm_best.pt when it improves
    (best model by val loss, never rotated away),
  - generates an AUDIO sample: continues a held-out prompt and decodes it through
    the codec, writing samples/gen_<step>.wav + samples/spec_<step>.png so you can
    hear/see progress (set audio_samples=false to skip).
Metrics (train + val) are appended to ckpt_dir/metrics.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from ..audio_io import save_audio
from ..data.token_dataset import TokenWindowDataset, list_token_files
from ..utils import (AverageMeter, amp_dtype, build_optimizer, count_params,
                     get_device, load_checkpoint, load_config, lr_lambda,
                     lr_lambda_cosine, rotate_checkpoints, save_checkpoint, set_seed)
from ..viz import save_spectrogram_compare
from .model import SpeechLM, loss_fn


@torch.no_grad()
def token_accuracy(logits, target, pad):
    pred = logits.argmax(-1)
    mask = target != pad
    return ((pred == target) & mask).sum().item() / max(mask.sum().item(), 1)


def latest_ckpt(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    cks = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("lm_") and f.endswith(".pt"):
            try:
                cks.append((int(f[3:-3]), f))      # numbered only; skip lm_best.pt
            except ValueError:
                continue
    return os.path.join(ckpt_dir, max(cks)[1]) if cks else None


@torch.no_grad()
def validate(model, val_dl, pad, device, dtype, use_amp, max_batches, agent_start):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for i, win in enumerate(val_dl):
        if i >= max_batches:
            break
        win = win.to(device, non_blocking=True)
        inp = win[:, :, :-1].transpose(1, 2)
        tgt = win[:, agent_start:, 1:].transpose(1, 2)
        with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
            logits = model(inp, tgt)
            loss = loss_fn(logits, tgt, pad)
        loss_m.update(loss.item())
        acc_m.update(token_accuracy(logits, tgt, pad))
    model.train()
    return loss_m.avg, acc_m.avg


@torch.no_grad()
def generate_audio_sample(model, codec, prompt_codes, gen_frames, device, step, sample_dir,
                          sr, temperature=1.0, top_k=250, top_p=0.95,
                          repetition_penalty=1.3, rep_window=50):
    """prompt_codes: (1, P, n_q) real tokens. Continues them, decodes prompt-only vs
    prompt+generated, and writes a wav + a spectrogram comparison.

    Defaults match the recommended generation knobs (temp 1.0 + repetition penalty),
    so checkpoint samples reflect the model's real free-running quality rather than the
    silence/drone that low-temperature greedy sampling collapses into."""
    model.eval()
    max_ctx = model.pos.size(1)
    codes = prompt_codes.to(device)
    for _ in range(gen_frames):
        ctx = codes[:, -max_ctx:]
        h_t = model.temporal_ctx(ctx)[:, -1]
        recent = codes[:, -rep_window:].transpose(1, 2) if repetition_penalty > 1 else None
        frame = model.depth_sample(h_t, temperature, top_k, top_p,
                                   repetition_penalty, recent)   # (1, n_q)
        codes = torch.cat([codes, frame.unsqueeze(1)], dim=1)
    gen_wav = codec.decode(codes).float().clamp(-1, 1).cpu()[0]
    prompt_wav = codec.decode(prompt_codes.to(device)).float().clamp(-1, 1).cpu()[0]
    save_audio(os.path.join(sample_dir, f"gen_{step}.wav"), gen_wav, sr)
    save_spectrogram_compare(prompt_wav, gen_wav,
                             os.path.join(sample_dir, f"spec_{step}.png"), sr)
    model.train()


def main():
    cfg = load_config("configs/lm.yaml")
    set_seed(cfg.seed)
    device = get_device()
    dtype = amp_dtype(cfg.precision)
    use_amp = device.type == "cuda" and cfg.precision != "fp32"
    pad = cfg.codebook_size
    n_q = cfg.n_q
    agent_start = (cfg.n_streams - 1) * n_q

    # train / val split over token files
    files = list_token_files(cfg.token_dir)
    rng = random.Random(cfg.seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * cfg.get("val_frac", 0.01)))
    val_files, train_files = files[:n_val], files[n_val:]

    train_ds = TokenWindowDataset(cfg.token_dir, cfg.window_frames, pad, files=train_files)
    val_ds = TokenWindowDataset(cfg.token_dir, cfg.window_frames, pad, files=val_files)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    model = SpeechLM(cfg.n_streams, cfg.n_q, cfg.codebook_size, cfg.d,
                     cfg.temporal_layers, cfg.depth_layers, cfg.heads,
                     max_frames=cfg.window_frames + 8,
                     grad_checkpointing=cfg.get("grad_checkpointing", False)).to(device)
    print(f"{len(train_ds)} train / {len(val_ds)} val files | "
          f"SpeechLM {count_params(model)/1e6:.1f}M params | streams {cfg.n_streams} "
          f"| window {cfg.window_frames}")

    opt = build_optimizer(model.parameters(), cfg.lr, cfg.get("optimizer", "adamw"))
    if cfg.get("lr_schedule", "cosine") == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_lambda_cosine(s, cfg.warmup, cfg.max_steps,
                                            cfg.get("lr_min_ratio", 0.1)))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup))

    step, best_val = 0, float("inf")
    # resume_from (e.g. checkpoints/lm/lm_best.pt) overrides auto-resume from the latest
    # checkpoint — use it to restart from a known-good point after a divergence.
    ckpt = cfg.get("resume_from") or latest_ckpt(cfg.ckpt_dir)
    if ckpt and os.path.exists(ckpt):
        blob = load_checkpoint(ckpt, map_location=device, model=model, opt=opt, sched=sched)
        step = blob.get("step", 0)
        best_val = blob.get("best_val", float("inf"))
        print(f"resumed from {ckpt} at step {step} (best_val {best_val:.4f})")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(cfg.ckpt_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    jsonl = open(os.path.join(cfg.ckpt_dir, "metrics.jsonl"), "a")

    # optional codec, loaded once, for audio samples (decode only)
    codec = None
    audio_on = cfg.get("audio_samples", True)
    prompt_codes = None
    if audio_on:
        try:
            from ..codec.model import load_codec
            codec = load_codec(cfg.codec_ckpt, device)
            import numpy as np
            P = cfg.get("sample_prompt_frames", 100)
            vc = np.load(val_files[0])[:n_q]                 # (n_q, T) take agent stream
            prompt_codes = torch.from_numpy(vc[:, :P].T.astype("int64")).unsqueeze(0)
            print(f"audio samples on | codec @ {codec.frame_rate:.1f} Hz | prompt {P} frames")
        except Exception as e:
            print(f"[warn] audio samples disabled: {e}")
            audio_on = False

    model.train()
    loss_m, acc_m, t0 = AverageMeter(), AverageMeter(), time.time()
    data_iter = iter(dl)
    last_logits = last_tgt = None

    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            try:
                win = next(data_iter)
            except StopIteration:
                data_iter = iter(dl); win = next(data_iter)
            win = win.to(device, non_blocking=True)
            inp = win[:, :, :-1].transpose(1, 2)
            tgt = win[:, agent_start:, 1:].transpose(1, 2)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(inp, tgt)
                loss = loss_fn(logits, tgt, pad) / cfg.grad_accum
            loss.backward()
            loss_m.update(loss.item() * cfg.grad_accum)
            last_logits, last_tgt = logits, tgt

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # skip the update if the step went non-finite, so one bad batch can't corrupt
        # the weights (a sustained divergence still needs the LR schedule / lower LR)
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            print(f"[warn] non-finite grad at step {step+1}; skipping update")
        sched.step(); step += 1
        acc_m.update(token_accuracy(last_logits, last_tgt, pad))

        if step % cfg.log_every == 0:
            sps = cfg.log_every / (time.time() - t0); t0 = time.time()
            row = {"step": step, "loss": round(loss_m.avg, 4),
                   "ppl": round(math.exp(min(loss_m.avg, 20)), 1),
                   "acc": round(acc_m.avg, 4), "lr": sched.get_last_lr()[0],
                   "it_per_s": round(sps, 3)}
            jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
            print(f"step {step:>7} | loss {row['loss']:.4f} | ppl {row['ppl']:.1f} "
                  f"| acc {row['acc']*100:.1f}% | lr {row['lr']:.2e} | {sps:.2f} it/s")
            loss_m.reset(); acc_m.reset()

        if step % cfg.ckpt_every == 0:
            val_loss, val_acc = validate(model, val_dl, pad, device, dtype, use_amp,
                                         cfg.get("val_batches", 20), agent_start)
            vrow = {"step": step, "val_loss": round(val_loss, 4),
                    "val_ppl": round(math.exp(min(val_loss, 20)), 1),
                    "val_acc": round(val_acc, 4)}
            jsonl.write(json.dumps(vrow) + "\n"); jsonl.flush()
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            print(f"  [val] step {step} | val_loss {val_loss:.4f} "
                  f"| val_ppl {vrow['val_ppl']:.1f} | val_acc {val_acc*100:.1f}% "
                  f"| best {best_val:.4f}{'  <- new best' if is_best else ''}")

            path = os.path.join(cfg.ckpt_dir, f"lm_{step}.pt")
            save_checkpoint(path, step, model=model, opt=opt, sched=sched,
                            best_val=best_val, config=dict(cfg))
            if is_best:
                save_checkpoint(os.path.join(cfg.ckpt_dir, "lm_best.pt"), step,
                                model=model, opt=opt, sched=sched,
                                best_val=best_val, config=dict(cfg))
            rotate_checkpoints(cfg.ckpt_dir, "lm", cfg.get("keep_last", 2))
            print("saved", path)

            if audio_on:
                try:
                    gf = cfg.get("sample_gen_frames", int(3 * codec.frame_rate))
                    generate_audio_sample(
                        model, codec, prompt_codes, gf, device, step, sample_dir,
                        codec.sample_rate,
                        temperature=cfg.get("gen_temperature", 1.0),
                        top_k=cfg.get("gen_top_k", 250),
                        top_p=cfg.get("gen_top_p", 0.95),
                        repetition_penalty=cfg.get("gen_repetition_penalty", 1.3),
                        rep_window=cfg.get("gen_rep_window", 50))
                except Exception as e:
                    print(f"[warn] audio sample failed at step {step}: {e}")

    jsonl.close()
    print("training complete")


if __name__ == "__main__":
    main()