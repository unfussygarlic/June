"""Train the textless speech LM on precomputed codec tokens.

    python -m s2s.lm.train_lm --config configs/lm.yaml
    python -m s2s.lm.train_lm --config configs/lm.yaml n_streams=2   # dialogue

Next-frame prediction: given frames up to t, predict frame t+1's K codebooks.
"""
from __future__ import annotations

import math
import os
import time

import torch
from torch.utils.data import DataLoader

from ..data.token_dataset import TokenWindowDataset
from ..utils import (AverageMeter, amp_dtype, build_optimizer, count_params,
                     get_device, load_checkpoint, load_config, lr_lambda,
                     save_checkpoint, set_seed)
from .model import SpeechLM, loss_fn


def latest_ckpt(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    return os.path.join(ckpt_dir, max(cks, key=lambda f: int(f.split("_")[-1][:-3]))) if cks else None


def main():
    cfg = load_config("configs/lm.yaml")
    set_seed(cfg.seed)
    device = get_device()
    dtype = amp_dtype(cfg.precision)
    use_amp = device.type == "cuda" and cfg.precision != "fp32"
    pad = cfg.codebook_size

    ds = TokenWindowDataset(cfg.token_dir, cfg.window_frames, pad)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                    persistent_workers=cfg.num_workers > 0)

    model = SpeechLM(cfg.n_streams, cfg.n_q, cfg.codebook_size, cfg.d,
                     cfg.temporal_layers, cfg.depth_layers, cfg.heads,
                     max_frames=cfg.window_frames + 8,
                     grad_checkpointing=cfg.get("grad_checkpointing", False)).to(device)
    print(f"{len(ds)} token files | SpeechLM {count_params(model)/1e6:.1f}M params "
          f"| streams {cfg.n_streams} | window {cfg.window_frames}")

    opt = build_optimizer(model.parameters(), cfg.lr, cfg.get("optimizer", "adamw"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg.warmup))

    step = 0
    ckpt = latest_ckpt(cfg.ckpt_dir)
    if ckpt:
        blob = load_checkpoint(ckpt, map_location=device, model=model, opt=opt, sched=sched)
        step = blob.get("step", 0)
        print(f"resumed from {ckpt} at step {step}")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    model.train()
    loss_m, t0 = AverageMeter(), time.time()
    data_iter = iter(dl)
    n_q = cfg.n_q
    agent_start = (cfg.n_streams - 1) * n_q          # first agent channel

    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            try:
                win = next(data_iter)
            except StopIteration:
                data_iter = iter(dl); win = next(data_iter)
            win = win.to(device, non_blocking=True)              # (B, C, W)

            inp = win[:, :, :-1].transpose(1, 2)                 # (B, W-1, C)
            tgt = win[:, agent_start:, 1:].transpose(1, 2)       # (B, W-1, n_q)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                logits = model(inp, tgt)
                loss = loss_fn(logits, tgt, pad) / cfg.grad_accum
            loss.backward()
            loss_m.update(loss.item() * cfg.grad_accum)

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step(); step += 1

        if step % cfg.log_every == 0:
            sps = cfg.log_every / (time.time() - t0); t0 = time.time()
            ppl = math.exp(min(loss_m.avg, 20))
            print(f"step {step:>7} | loss {loss_m.avg:.4f} | ppl {ppl:.1f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | {sps:.1f} it/s")
            loss_m.reset()

        if step % cfg.ckpt_every == 0:
            path = os.path.join(cfg.ckpt_dir, f"lm_{step}.pt")
            save_checkpoint(path, step, model=model, opt=opt, sched=sched, config=dict(cfg))
            print("saved", path)

    print("training complete")


if __name__ == "__main__":
    main()