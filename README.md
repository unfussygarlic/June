# June

A **textless speech-to-speech** language model, built from scratch and trained on a
single consumer GPU. Audio in, audio out — no text and no pretrained LLM anywhere in
the loop. June learns the sound of language (phonetics, lexicon, prosody, rhythm)
directly from raw speech, in the lineage of GSLM → dGSLM → Moshi, scaled down to fit
an **RTX 4060 (8 GB)**.

This is a learning/research project. The point was to build and understand every
component end to end, not to match a production system — and to do it on hardware
most people wouldn't attempt it on.

## What it actually does (honest results)

June takes a short spoken prompt and **continues it** — generating speech that keeps
going in a plausible voice. The output is **fluent-sounding but robotic and
semantically meaningless**: real English words in locally-plausible sequences that
don't add up to actual sentences. Like overhearing confident speech in a language you
half-know — the melody and words are right, the meaning slips away.

Two honest limitations, each with a clear cause:

- **Robotic voice** — the fidelity ceiling of the small (~7M) from-scratch codec. The
  codec *is* the voice; even perfect tokens decoded through it sound rough.
- **No meaning** — the LM learned form (how English sounds) but not semantics. Real
  meaning needs far more data, model, and compute than a 4060 provides. This is the
  genuine research frontier (what Dessa used 72 GB200s for).

Both are expected outcomes at this scale. June is a complete, working pipeline that
reaches the realistic ceiling of 127M params / ~460h / 8 GB.

### Final numbers

| Component | Result |
| --- | --- |
| **Codec** (~7M params, 50 Hz, 8×1024 RVQ) | resynthesis **WER 10.7%** (Whisper round-trip) |
| **LM** (127M: d=768, 12 temporal + 4 depth layers) | val **perplexity ~19**, val token-accuracy **~23%** |
| Training data | LibriSpeech train-clean-100 + train-clean-360 (**~460h**) |
| Generation speechiness (Whisper) | avg_logprob −0.83, no_speech_prob 0.09, ~3 words/s |
| Hardware | one RTX 4060 8 GB, ~55 W power cap, ~0.34 it/s at d=768 |

## Repo layout

```
june/  (python package: s2s)
├── configs/
│   ├── codec.yaml           codec hyperparameters
│   └── lm.yaml              LM hyperparameters, sampling + eval knobs
├── s2s/
│   ├── audio_io.py          soundfile I/O (no torchcodec/ffmpeg needed)
│   ├── utils.py             config, checkpoints (+ rotation), schedules, optimizers
│   ├── smoke_test.py        CPU shape/gradient checks — run first
│   ├── viz.py               spectrogram comparison PNGs
│   ├── data/
│   │   ├── download.py      resumable LibriSpeech downloader (survives flaky nets)
│   │   ├── audio_dataset.py random audio crops for codec training
│   │   ├── tokenize.py      encode a corpus to token shards
│   │   ├── tokenize_tar.py  encode straight from a .tar.gz, no extraction (saves disk)
│   │   └── token_dataset.py windowed token batches for LM training
│   ├── codec/
│   │   ├── quantizer.py     L2-normalized factorized RVQ (EMA + restart; anti-collapse)
│   │   ├── seanet.py        conv encoder / decoder
│   │   ├── discriminators.py HiFi-GAN multi-period + multi-scale
│   │   ├── losses.py        multi-res mel, LSGAN adversarial, feature matching
│   │   ├── model.py         NeuralCodec = encoder + RVQ + decoder
│   │   └── train_codec.py   GAN training (bf16, resumable, codebook-usage logging)
│   ├── lm/
│   │   ├── model.py         SpeechLM (temporal + depth transformer), sampling, loaders
│   │   ├── delay.py         delay-pattern utilities (alternative multi-codebook path)
│   │   ├── streaming.py     KV-cache for faster-than-offline streaming generation
│   │   ├── train_lm.py      training: cosine LR, validation, best-model, audio samples
│   │   ├── sample.py        offline continuation from a seed clip
│   │   └── sample_stream.py streaming continuation + real-time-factor report
│   └── eval/
│       ├── resynth_wer.py   codec content-preservation (Whisper WER)
│       └── lm_eval.py       LM val perplexity + Whisper word-likeness metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# install the CUDA build of torch for YOUR GPU, e.g.:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# optional extras:
pip install bitsandbytes      # 8-bit optimizer (adam8bit) — needed for the 127M LM on 8 GB
pip install openai-whisper wordfreq   # evaluation metrics
```

Install notes learned the hard way:
- **Audio I/O goes through `soundfile` (libsndfile), not `torchaudio.load/save`** —
  recent torchaudio routes I/O through a separate `torchcodec` package and errors
  without it. If `pip install soundfile` doesn't pull libsndfile on your platform:
  `apt install libsndfile1` / `brew install libsndfile`.
- Evaluation feeds Whisper audio arrays directly, so **ffmpeg is not required**.

Run everything as a module from the repo root (`python -m s2s....`). Any config value
can be overridden on the command line: `... batch_size=8 lr=1e-4`.

## The full pipeline

```bash
# 0. sanity check (CPU, ~10s, no data needed)
python -m s2s.smoke_test

# 1. data (resumable — safe to re-run if your connection drops)
python -m s2s.data.download --config configs/codec.yaml                 # train-clean-100
python -m s2s.data.download --config configs/codec.yaml subset=dev-clean

# 2. train the codec (do this first — nothing downstream works without it)
python -m s2s.codec.train_codec --config configs/codec.yaml
#   watch checkpoints/codec/samples/*.png and the cb_active count in the logs

# 3. tokenize the corpus once (frees the codec + audio from VRAM afterwards)
python -m s2s.data.tokenize --config configs/lm.yaml \
    codec_ckpt=checkpoints/codec/codec_22000.pt
#   low on disk? tokenize straight from the archive instead, then delete it:
#   python -m s2s.data.tokenize_tar --config configs/lm.yaml \
#       archive=data/train-clean-360.tar.gz subset=train-clean-360

# 4. train the LM
python -m s2s.lm.train_lm --config configs/lm.yaml
#   it validates, saves lm_best.pt on val improvement, and writes audio samples
#   + metrics.jsonl at every checkpoint

# 5. generate / listen
python -m s2s.lm.sample_stream --config configs/lm.yaml \
    lm_ckpt=checkpoints/lm/lm_best.pt \
    seed_wav=data/LibriSpeech/dev-clean/<...>.flac \
    gen_seconds=6 out=output.wav
```

## Inference

June is a **continuation** model — it needs a spoken prompt to continue (not
text-to-speech). The first ~2 s of the output is the seed; the rest is generated.

```bash
# streaming (KV-cached, prints real-time factor)
python -m s2s.lm.sample_stream --config configs/lm.yaml \
    lm_ckpt=checkpoints/lm/lm_best.pt \
    seed_wav=<any 16kHz-ish speech clip> \
    gen_seconds=6 temperature=1.0 top_k=250 top_p=0.95 \
    repetition_penalty=1.3 rep_window=50 out=output.wav
```

Sampling knobs that matter:
- `temperature` 1.0 + `repetition_penalty` 1.3 — the combo that gives fluent output.
  Low temperature alone collapses to silence/drone; the penalty prevents it.
- Drop to `temperature=0.7` for more conservative (risks mumbling); raise toward 1.1
  for more varied.
- If it drones or loops, raise `repetition_penalty` to 1.4; if it's frantic, drop to 1.2.

The **streaming** path uses a KV cache for the temporal transformer — verified to
produce bit-identical output to the offline path, and faster (O(T) per frame instead
of O(T²) per clip). Context is capped at the model's positional length.

## Evaluation

```bash
# codec: does encode->decode preserve the words?
python -m s2s.eval.resynth_wer --config configs/codec.yaml \
    codec_ckpt=checkpoints/codec/codec_22000.pt n_files=50

# LM: val perplexity + how speech/word-like the generations are (runs on CPU)
python -m s2s.eval.lm_eval --config configs/lm.yaml \
    lm_ckpt=checkpoints/lm/lm_best.pt eval_clips=6 device=cpu
```
`lm_eval` reports val perplexity, words-per-second, Whisper's avg_logprob and
no_speech_prob, and (with `wordfreq`) a real-word rate. Run it across checkpoints to
trace the babble → words progression. Note: Whisper snaps marginal audio to real
words, so read `real_word_rate` alongside `avg_logprob`, not alone.

## Fitting 8 GB

- Codec OOM: `batch_size=8 crop_seconds=0.6`.
- LM OOM: `window_frames=375`, then `batch_size=2 grad_accum=16`, then `d=384`.
- `precision=bf16` default; `optimizer=adam8bit` (bitsandbytes) shaves optimizer VRAM.
- The 127M LM at d=768 fits ~6.3 GB at `batch_size=4 window_frames=375` without
  gradient checkpointing; d=1024 (220M) needs `grad_checkpointing=true`.
- Both trainers auto-resume from the latest checkpoint; `resume_from=<path>` overrides
  (e.g. to restart from `lm_best.pt` after a divergence).

## Lessons learned

The debugging was the education. Three things worth recording:

1. **Codebook collapse** (codec). Plain euclidean VQ collapsed to ~25% codebook use
   with the commitment loss exploding. Fix: DAC-style **L2-normalized factorized
   codes** (quantize on the unit sphere in a low-dim space) + dead-code restart +
   fp32 EMA. Usage recovered and stayed healthy.

2. **Training divergence** (LM), twice. The loss trained smoothly then blew up
   catastrophically — first at ~15k steps (constant LR 3e-4), then again at ~38k even
   after switching to **cosine decay + peak 2e-4** (LR was still ~1.8e-4 there). The
   stability threshold for sustained training sits below ~1.8e-4. The non-finite-grad
   guard doesn't catch it because it's a slow drift, not a spike. **Validation +
   best-model checkpointing is what saved both runs** — `lm_best.pt` always held the
   pre-divergence peak. If pushing further: peak LR ~1e-4, tighter grad clip, fresh
   optimizer state on data changes.

3. **Data ceiling.** On 100h the LM plateaued (val_acc stuck ~20.7%, train/val gap
   widening = overfitting). Adding train-clean-360 (→460h) immediately broke through
   (val_acc 20.7% → 23.4%, gap flipped negative). Confirmed empirically that the
   bottleneck was data, not model capacity. 460h then plateaued ~23.4% — the next
   ceiling, needing train-other-500 / LibriLight to move.

## Going further

- **Better voice:** swap the from-scratch codec for a pretrained **Mimi / EnCodec /
  DAC** (far higher fidelity), then re-tokenize and retrain the LM on its tokens.
  This is the real fix for the robotic quality.
- **12.5 Hz codec** (Mimi-like, 4× shorter sequences): `strides=[2,4,5,8,4]` +
  `frame_rate=12.5`, retrain codec, re-tokenize.
- **More data:** train-other-500, LibriLight.
- **Two-stream dialogue** (the full dGSLM/Dessa goal): token shards of shape
  `(2*n_q, T)` — user then agent codebooks on one clock — then `n_streams=2`. The
  architecture already supports it; two-channel conversational data is the bottleneck.

## References

GSLM / dGSLM (textless speech & dialogue LMs) · SoundStream / EnCodec / DAC (neural
audio codecs) · SpeechTokenizer & Mimi (semantic-distilled RVQ) · AudioLM & Moshi
(temporal + depth transformer over codec tokens) · MusicGen (delay pattern) ·
HiFi-GAN (discriminators). See each project for the original work.

## License

MIT — add a `LICENSE` file before publishing if you want it enforced. An educational
implementation; weights you train are your own.