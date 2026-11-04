"""Evaluate a textless speech LM: validation perplexity + how speech/word-like its
free-running generations are. Tracks the babble -> words progression that loss alone
can't show.

    pip install openai-whisper
    python -m s2s.eval.lm_eval --config configs/lm.yaml \
        lm_ckpt=checkpoints/lm/lm_best.pt eval_clips=6 device=cpu

Runs on CPU by default so it doesn't fight a training job for the GPU. Reports:
  - val_ppl           : token perplexity on the held-out split (the model's own metric)
  - words, words_per_s: how much transcribable speech Whisper finds in the generations
  - avg_logprob       : Whisper's mean token log-prob (higher = more speech-like)
  - no_speech_prob    : Whisper's mean P(not speech) (lower = more speech-like)
  - real_word_rate    : fraction of transcribed words that are real English words
                        (only if `wordfreq` is installed; skipped otherwise)
Run it on a few checkpoints over training and the numbers trace the model learning
to actually say words.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random

import numpy as np
import torch

from ..audio_io import save_audio
from ..codec.model import load_codec
from ..data.token_dataset import list_token_files
from ..lm.model import load_speechlm
from ..lm.streaming import generate_stream


def val_perplexity(model, token_dir, window, n_q, pad, device, n_batches=20, bs=4):
    from ..data.token_dataset import TokenWindowDataset
    from torch.utils.data import DataLoader
    from ..lm.model import loss_fn
    files = list_token_files(token_dir)
    random.Random(0).shuffle(files)
    val = files[:max(1, len(files) // 100)]
    dl = DataLoader(TokenWindowDataset(token_dir, window, pad, files=val),
                    batch_size=bs, shuffle=False)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, win in enumerate(dl):
            if i >= n_batches:
                break
            win = win.to(device)
            inp = win[:, :, :-1].transpose(1, 2)
            tgt = win[:, :, 1:].transpose(1, 2)
            tot += loss_fn(model(inp, tgt), tgt, pad).item()
            n += 1
    return math.exp(tot / max(n, 1))


def speechiness(wav_paths, asr):
    """Transcribe each clip and aggregate Whisper's speech-likeness signals.

    Loads audio with soundfile and passes a numpy array to Whisper, so it does NOT
    shell out to ffmpeg (which may not be installed). The clips are already 16 kHz
    mono, which is what Whisper expects."""
    import numpy as np
    import soundfile as sf
    words, logp, nosp, secs, texts = 0, [], [], 0.0, []
    for p in wav_paths:
        data, _ = sf.read(p, dtype="float32", always_2d=True)      # (L, C)
        audio = data.mean(1)                                        # mono 1-D float32
        secs += len(audio) / 16000
        r = asr.transcribe(audio, language="en")
        txt = r["text"].strip()
        texts.append(txt)
        words += len(txt.split())
        for s in r.get("segments", []):
            logp.append(s.get("avg_logprob", -5))
            nosp.append(s.get("no_speech_prob", 1.0))
    out = {
        "words": words,
        "words_per_s": round(words / max(secs, 1e-6), 3),
        "avg_logprob": round(float(np.mean(logp)) if logp else -5, 3),
        "no_speech_prob": round(float(np.mean(nosp)) if nosp else 1.0, 3),
        "sample_texts": texts[:3],
    }
    try:
        from wordfreq import zipf_frequency
        allw = [w.strip(".,!?;:'\"").lower() for t in texts for w in t.split()]
        allw = [w for w in allw if w]
        real = sum(zipf_frequency(w, "en") > 0 for w in allw)
        out["real_word_rate"] = round(real / max(len(allw), 1), 3)
    except ImportError:
        out["real_word_rate"] = None       # pip install wordfreq to enable
    return out


def main():
    cfg = load_config_eval()
    device = torch.device(cfg.get("device", "cpu"))
    codec = load_codec(cfg["codec_ckpt"], device)
    model, mcfg = load_speechlm(cfg["lm_ckpt"], device)
    n_q = mcfg.get("n_q", 8)
    pad = mcfg.get("codebook_size", 1024)
    sr = codec.sample_rate

    print(f"evaluating {cfg['lm_ckpt']} on {device}")
    ppl = val_perplexity(model, cfg["token_dir"], mcfg.get("window_frames", 500),
                         n_q, pad, device, n_batches=cfg.get("val_batches", 20))
    print(f"  val_ppl: {ppl:.2f}")

    # generate a few continuations from held-out prompts, decode, transcribe
    out_dir = cfg.get("eval_out", "eval_samples")
    os.makedirs(out_dir, exist_ok=True)
    val_files = sorted(glob.glob(os.path.join(cfg["token_dir"], "**", "*.npy"), recursive=True))
    random.Random(1).shuffle(val_files)
    n_clips = cfg.get("eval_clips", 6)
    P = cfg.get("sample_prompt_frames", 100)
    G = int(cfg.get("gen_seconds", 4) * codec.frame_rate)

    wav_paths = []
    for i in range(n_clips):
        vc = np.load(val_files[i])[:n_q]                       # (n_q, T)
        prompt = torch.from_numpy(vc[:, :P].T.astype("int64")).unsqueeze(0).to(device)
        codes = generate_stream(model, prompt, G)
        wav = codec.decode(codes).float().clamp(-1, 1).cpu()[0]
        wp = os.path.join(out_dir, f"eval_{i}.wav")
        save_audio(wp, wav, sr)
        wav_paths.append(wp)

    try:
        import whisper
        asr = whisper.load_model("base.en")
        stats = speechiness(wav_paths, asr)
    except ImportError:
        print("  (pip install openai-whisper for word-likeness metrics)")
        stats = {}

    report = {"lm_ckpt": cfg["lm_ckpt"], "val_ppl": round(ppl, 2), **stats}
    print(json.dumps(report, indent=2))
    with open(os.path.join(out_dir, "eval.jsonl"), "a") as f:
        f.write(json.dumps(report) + "\n")


def load_config_eval() -> dict:
    """Read lm.yaml and apply key=value CLI overrides (same convention as utils)."""
    import argparse
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lm.yaml")
    args, extra = ap.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    for ov in extra:
        k, v = ov.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cfg[k] = v
    assert "lm_ckpt" in cfg, "pass lm_ckpt=checkpoints/lm/lm_best.pt"
    return cfg


if __name__ == "__main__":
    main()