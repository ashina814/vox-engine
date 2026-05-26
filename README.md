# VOX

[![lint](https://github.com/ashina814/vox-engine/actions/workflows/lint.yml/badge.svg)](https://github.com/ashina814/vox-engine/actions/workflows/lint.yml)
[![test](https://github.com/ashina814/vox-engine/actions/workflows/test.yml/badge.svg)](https://github.com/ashina814/vox-engine/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-136%20fast%20%2B%203%20slow-brightgreen.svg)](#tests)

**Singing Voice Conversion engine — SVC-first, SVS-ready.**

VOX takes a vocal track and re-synthesises it in a target singer's voice
(and optional style), preserving phrasing while exposing user-controlled
musical knobs (auto-tune, style blend).

## Status

Phase A (pre-grant) is **feature-complete** — every component from the
design spec is implemented, tested, and runnable without a GPU:

| Block | Status | Tests |
|---|---|---|
| A1 Repo, CI, pyproject | ✅ | — |
| A2 Data pipeline (mel/f0/uv/content/loudness, chunking, QA, Dataset, preprocess CLI) | ✅ | 37 |
| A3 Models (Aggregator, GST, Whisper branch, DiffusionDecoder, Vocoder wrapper, VoxModel) | ✅ | 29 |
| A4 Training (Trainer, losses, optim, logging, ckpt) | ✅ | 17 |
| A5 Inference (Pipeline, Auto-Tune, Style blend, CLI) | ✅ | 27 |
| A6 Gradio GUI | ✅ | 3 |
| A7 Evaluation (MCD, F0 RMSE, UV err, Style separability, Benchmark) | ✅ | 23 |
| A8 Tests | ✅ (136 fast + 3 slow) | — |
| A9 Docs | ✅ | — |
| OpenSinger baseline training (MA-5) | pending | — |

Phase B (post-grant) is **"just train the model"** — recording, fine-tuning
the NSF-HiFiGAN on the recorded mels, and running the production training
schedule are the only remaining steps.

## Setup

```bash
uv sync                    # runtime deps
uv sync --extra dev        # + pytest, ruff, mypy, black
```

Python 3.11. No Rust toolchain required (all native deps ship as wheels).

## Quick start

```bash
# 1. Preprocess a dataset (drops raw wavs under dataset/raw/{style}/)
uv run python scripts/preprocess.py data=opensinger

# 2. Convert one vocal — using an untrained model just to verify wiring
uv run python scripts/infer.py input_wav=vocal.wav output_wav=vox.wav \
    +inference.skip_content=true inference.num_diffusion_steps=10

# 3. Launch the GUI
uv run python scripts/ui.py +ui.skip_content=true

# 4. Benchmark against a val split
uv run python scripts/benchmark.py data=opensinger +ckpt_path=ckpts/last.pt
```

Training itself runs through `vox.training.VoxTrainer` — call it from a
notebook today; `scripts/train.py` lands alongside the production training
run.

## Architecture (at a glance)

```
WAV → [ContentVec, F0, UV, Loudness] → Aggregator (GST + FiLM)
    → WhisperAwareConditioning → DiffusionDecoder (v-pred, K=50 DDIM)
    → NSF-HiFiGAN → WAV
```

Full design notes: [docs/architecture.md](docs/architecture.md).

## Documentation

| File | Topic |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module map, tensor conventions, design decisions |
| [docs/data_pipeline.md](docs/data_pipeline.md) | Feature extractors, chunking, QA, Dataset, preprocess CLI |
| [docs/training.md](docs/training.md) | Trainer, losses, optimizer, logging, checkpoints |
| [docs/inference.md](docs/inference.md) | InferencePipeline, Auto-Tune, Style blend, CLI / GUI |

## Tests

```bash
uv run --no-sync pytest tests/ -m "not slow"   # ~10s
uv run --no-sync pytest tests/                  # incl. slow MA-3 loss-decrease test
```

Test fixtures are deliberately tiny (B=2, T=20-40, hidden=32) so the full
fast suite runs in single-digit seconds on CPU.

## License

Apache-2.0. See [LICENSE](LICENSE).
