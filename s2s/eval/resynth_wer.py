"""Codec sanity check: encode->decode a few clips and measure how much content survives.

    pip install openai-whisper jiwer
    python -m s2s.eval.resynth_wer --config configs/codec.yaml \
        codec_ckpt=checkpoints/codec/codec_200000.pt n_files=20

Reports WER between Whisper's transcript of the original vs the resynthesized audio.
Low WER = the tokens kept the words. (This tests the codec, not the LM.)
"""
from __future__ import annotations

import glob
import os

import torch

from ..audio_io import load_audio
from ..codec.model import load_codec
from ..utils import get_device, load_config


def main():
    cfg = load_config("configs/codec.yaml")
    device = get_device()

    try:
        import whisper
        import jiwer
    except ImportError:
        raise SystemExit("pip install openai-whisper jiwer")

    codec = load_codec(cfg.get("codec_ckpt", "checkpoints/codec/codec_200000.pt"), device)
    asr = whisper.load_model("base.en")

    audio_root = os.path.join(cfg.data_root, "LibriSpeech", cfg.get("eval_subset", "dev-clean"))
    files = sorted(glob.glob(os.path.join(audio_root, "**", "*.flac"), recursive=True))
    files = files[: int(cfg.get("n_files", 20))]
    if not files:
        raise SystemExit(f"no eval audio under {audio_root}")

    refs, hyps = [], []
    for path in files:
        wav, _ = load_audio(path, cfg.sample_rate)
        with torch.no_grad():
            recon, _, _ = codec(wav.unsqueeze(0).to(device))
        recon = recon.float().clamp(-1, 1).cpu()[0]

        orig_txt = asr.transcribe(wav.squeeze(0).numpy(), language="en")["text"]
        recon_txt = asr.transcribe(recon.squeeze(0).numpy(), language="en")["text"]
        refs.append(orig_txt.lower().strip())
        hyps.append(recon_txt.lower().strip())

    wer = jiwer.wer(refs, hyps)
    print(f"resynthesis WER over {len(files)} files: {wer*100:.1f}%")


if __name__ == "__main__":
    main()
