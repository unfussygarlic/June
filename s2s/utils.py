"""Shared utilities: config loading, checkpointing, seeding, meters."""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import yaml


class Config(dict):
    """Dict with attribute access, loaded from YAML. Nested dicts become Config too."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(default_path: str | None = None) -> Config:
    """Load a YAML config file named by --config, with optional `key=value` CLI overrides.

    Example:
        python -m s2s.codec.train_codec --config configs/codec.yaml batch_size=8 lr=1e-4
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_path)
    args, overrides = parser.parse_known_args()
    assert args.config, "pass --config path/to/config.yaml"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for ov in overrides:
        assert "=" in ov, f"override must be key=value, got {ov!r}"
        k, v = ov.split("=", 1)
        try:
            v = yaml.safe_load(v)  # parse numbers/bools/lists
        except Exception:
            pass
        cfg[k] = v
    return Config(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(precision: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_checkpoint(path: str, step: int, **states) -> None:
    """Save a checkpoint. States may be nn.Modules/optimizers (we store .state_dict())
    or plain values (stored as-is)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {"step": step}
    for name, obj in states.items():
        blob[name] = obj.state_dict() if hasattr(obj, "state_dict") else obj
    torch.save(blob, path)


def load_checkpoint(path: str, map_location="cpu", **modules):
    """Load a checkpoint; for every module passed in, load its state_dict in place.
    Returns the full blob (so you can read `step`, plain values, etc.)."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    for name, mod in modules.items():
        if name in blob and hasattr(mod, "load_state_dict"):
            mod.load_state_dict(blob[name])
    return blob


def rotate_checkpoints(ckpt_dir: str, prefix: str, keep_last: int) -> None:
    """Delete all but the newest `keep_last` numbered checkpoints <prefix>_<step>.pt.
    Only files whose suffix is an integer step are considered, so named checkpoints
    like <prefix>_best.pt are never rotated away. keep_last <= 0 disables rotation."""
    if keep_last is None or keep_last <= 0:
        return
    numbered = []
    for f in os.listdir(ckpt_dir):
        if f.startswith(prefix + "_") and f.endswith(".pt"):
            try:
                numbered.append((int(f[len(prefix) + 1:-3]), f))
            except ValueError:
                continue                       # skip lm_best.pt etc.
    numbered.sort()
    for _, f in numbered[:-keep_last]:
        try:
            os.remove(os.path.join(ckpt_dir, f))
        except OSError:
            pass


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def build_optimizer(params, lr: float, kind: str = "adamw", weight_decay: float = 0.01,
                    betas=(0.9, 0.95)):
    """AdamW by default; 'adam8bit' uses bitsandbytes if installed (saves VRAM)."""
    if kind == "adam8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            print("[warn] bitsandbytes not installed; falling back to AdamW.")
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def lr_lambda(step: int, warmup: int) -> float:
    """Linear warmup then constant. Simple and robust for from-scratch training."""
    return min(1.0, (step + 1) / max(warmup, 1))


def lr_lambda_cosine(step: int, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay from 1.0 down to `min_ratio`. Decaying the LR
    late is the main guard against the loss blowing up after a long stable run."""
    import math
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * prog))