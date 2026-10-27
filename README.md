# June

A small, from-scratch implementation of a **textless speech-to-speech** model — audio
in, audio out, with no text and no pretrained LLM anywhere in the loop. It's the
dGSLM → Moshi line of work (a neural audio codec + a transformer language model over
its tokens), scaled down to train on a single consumer GPU (built and tested targeting
an **RTX 4060, 8 GB**).

This is a learning/research codebase: the goal is that every component is readable and
hackable, not that it matches a production system. At this scale you can expect a model
that produces speech-like babble → syllables → occasional words, and (with the right
data) plausible conversational timing — not open-domain conversation, which needs
orders of magnitude more compute and data.

## What's inside

The Python package is `s2s` (so commands are `python -m s2s....`); the project/repo is
**June**.

```
june/
├── configs/
│   ├── codec.yaml          # codec hyperparameters
│   └── lm.yaml             # language-model hyperparameters
├── s2s/
│   ├── audio_io.py         # soundfile-based load/save (no torchcodec dependency)
│   ├── utils.py            # config loading, checkpoints, seeding, optimizers
│   ├── smoke_test.py       # CPU shape/gradient checks — run this first
│   ├── data/
│   │   ├── download.py     # fetch a LibriSpeech subset
│   │   ├── audio_dataset.py# random audio crops for codec training
│   │   ├── tokenize.py     # encode the corpus to token shards (run once)
│   │   └── token_dataset.py# windows of tokens for LM training
│   ├── codec/
│   │   ├── quantizer.py    # VectorQuantizer (EMA + straight-through + restart), ResidualVQ
│   │   ├── seanet.py       # convolutional encoder / decoder
│   │   ├── discriminators.py # HiFi-GAN multi-period + multi-scale discriminators
│   │   ├── losses.py       # multi-resolution mel, LSGAN adversarial, feature matching
│   │   ├── model.py        # NeuralCodec = encoder + RVQ + decoder
│   │   └── train_codec.py  # GAN training loop (bf16, resumable, dumps sample wavs)
│   ├── lm/
│   │   ├── model.py        # SpeechLM: temporal + depth transformer, loss, sampling
│   │   ├── delay.py        # delay-pattern utilities (alternative to the depth transformer)
│   │   ├── train_lm.py     # next-frame LM training (grad-accum, bf16, resumable)
│   │   └── sample.py       # generate a continuation from a seed clip
│   └── eval/
│       └── resynth_wer.py  # codec content-preservation check via Whisper
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Install the CUDA build of torch for YOUR machine (requirements.txt ships a generic one):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Two install notes worth knowing up front:

- **Audio I/O goes through `soundfile` (libsndfile), not `torchaudio.load/save`.** Recent
  torchaudio versions delegate I/O to a separate `torchcodec` package and error without
  it; `soundfile` sidesteps that and reads LibriSpeech's FLAC directly. If `pip install
  soundfile` doesn't pull libsndfile on your platform, install it via your package
  manager (`apt install libsndfile1`, `brew install libsndfile`).
- Everything runs as a module from the repo root (`python -m s2s....`). Any config value
  can be overridden on the command line: `... batch_size=8 lr=1e-4`.

## Quickstart

```bash
# 0. sanity check — tiny forward/backward on CPU, no data needed (~10s)
python -m s2s.smoke_test

# 1. data
python -m s2s.data.download --config configs/codec.yaml                 # train-clean-100 (~6 GB)
python -m s2s.data.download --config configs/codec.yaml subset=dev-clean # for eval

# 2. train the codec (do this first — nothing downstream works without it)
python -m s2s.codec.train_codec --config configs/codec.yaml

# 3. tokenize the corpus once (frees the codec + audio from VRAM afterwards)
python -m s2s.data.tokenize --config configs/lm.yaml \
    codec_ckpt=checkpoints/codec/codec_50000.pt

# 4. train the language model
python -m s2s.lm.train_lm --config configs/lm.yaml

# 5. generate a continuation from a seed clip
python -m s2s.lm.sample --config configs/lm.yaml \
    seed_wav=data/LibriSpeech/dev-clean/<...>.flac \
    lm_ckpt=checkpoints/lm/lm_20000.pt gen_seconds=5 out=out.wav
```

While the **codec** trains, listen to `checkpoints/codec/samples/recon_*.wav` against
`orig_*.wav`, and watch the `cb0` number in the logs — that's how many codebook-0 entries
are in use. If it collapses toward single digits you have codebook collapse (the restart
logic fights it; lower the learning rate if it persists). While the **LM** trains, a
falling loss / perplexity means it's learning the token statistics.

## Fitting 8 GB

- Defaults target ~8 GB. If the **codec** OOMs: `batch_size=8 crop_seconds=0.6`.
- If the **LM** OOMs: `window_frames=250`, then `batch_size=2 grad_accum=16`, then `d=384`.
- `precision=bf16` is the default (the 4060 supports it). Set `optimizer=adam8bit`
  (install `bitsandbytes`) to shave optimizer memory further.
- Both trainers auto-resume from the latest checkpoint in their `ckpt_dir`.

## Going further

- **12.5 Hz codec** (shorter sequences, Mimi-like): set `strides=[2,4,5,8,4]` in
  `codec.yaml` and `frame_rate=12.5` in `lm.yaml`, then retrain the codec and re-tokenize.
- **Two-stream dialogue** (full-duplex): write token shards of shape `(2*n_q, T)` — user
  codebooks then agent codebooks on one shared clock — into a token dir, then run
  `python -m s2s.lm.train_lm --config configs/lm.yaml n_streams=2`.
- **Skip the from-scratch codec**: tokenize with a pretrained Mimi/EnCodec instead. The
  LM code is unchanged; only the encoder in `tokenize.py` swaps. This is the fastest way
  to start LM experiments.

## Design notes

- The codec's architecture is read from the **checkpoint** (not re-read from YAML) by
  `tokenize.py` and `sample.py`, so changing codec hyperparameters never silently
  mismatches the trained weights.
- EMA/codebook statistics are computed in fp32 even under bf16 autocast (running averages
  need the mantissa bits).
- Correctness (shapes, the straight-through gradient, config/CLI/data wiring) is covered
  by `s2s.smoke_test` and a synthetic end-to-end run. Training *dynamics* — whether the
  codec converges to clean audio and the LM starts forming words — are what you observe
  on real speech on your own GPU.

## References

The design follows, and is a good companion to, these papers and repos:
GSLM / dGSLM (textless speech & dialogue LMs), SoundStream / EnCodec / DAC (neural audio
codecs), SpeechTokenizer & Mimi (semantic-distilled RVQ), AudioLM & Moshi (temporal +
depth transformer over codec tokens), MusicGen (delay pattern), and HiFi-GAN (the
discriminators). See each project for the original work.

## License

MIT — add a `LICENSE` file before publishing if you want it enforced. This is an
educational implementation; model weights you train are your own.