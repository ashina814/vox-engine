# VOX

[![lint](https://github.com/ashina814/vox-engine/actions/workflows/lint.yml/badge.svg)](https://github.com/ashina814/vox-engine/actions/workflows/lint.yml)
[![test](https://github.com/ashina814/vox-engine/actions/workflows/test.yml/badge.svg)](https://github.com/ashina814/vox-engine/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-161%20fast%20%2B%203%20slow-brightgreen.svg)](#tests)

**Singing Voice Conversion engine — SVC-first, SVS-ready.**

VOX takes a vocal track and re-synthesises it in a target singer's voice
(and optional style), preserving phrasing while exposing user-controlled
musical knobs (auto-tune, style blend).

> ⚠️ Voice synthesis is dual-use technology. Before using or
> redistributing this engine, read
> **[ACCEPTABLE_USE.md](ACCEPTABLE_USE.md)** and
> **[MODEL_CARD.md](MODEL_CARD.md)**. In short: only clone voices with
> the speaker's documented consent, label AI-generated output, and don't
> bypass voice authentication systems.

## Status

Phase A (pre-grant) is **feature-complete** — every component from the
design spec is implemented, tested, and runnable without a GPU:

| Block | Status | Tests |
|---|---|---|
| A1 Repo, CI, pyproject | ✅ | — |
| A2 Data pipeline (mel/f0/uv/content/loudness, chunking, QA, Dataset, preprocess CLI) | ✅ | 37 |
| A3 Models (Aggregator, GST, Whisper branch, Diffusion / Flow Matching decoder, Vocoder wrapper, VoxModel) | ✅ | 36 |
| A4 Training (Trainer, losses, optim, logging, ckpt, EMA, bf16) | ✅ | 25 |
| A5 Inference (Pipeline, Auto-Tune, Style blend, CLI) | ✅ | 27 |
| A6 Gradio GUI | ✅ | 3 |
| A7 Evaluation (MCD, F0 RMSE, UV err, Style separability, Benchmark) | ✅ | 23 |
| A8 Tests | ✅ (161 fast + 2 CUDA-skip + 3 slow) | — |
| A9 Docs | ✅ | — |
| OpenSinger baseline training (MA-5) | pending (Phase B) | — |

Phase B (post-grant) records ~5.5h of own singing, fine-tunes the
NSF-HiFiGAN vocoder, runs the production training schedule, releases 3
tracks on streaming services, and publishes a technical write-up.

## Setup

```bash
uv sync                    # runtime deps
uv sync --extra dev        # + pytest, ruff, mypy, black
```

Python 3.11. No Rust toolchain required (all native deps ship as wheels).
For CUDA, swap to the matching PyTorch wheel:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
    --reinstall torch torchaudio
```

## Quick start

```bash
# 1. Download external pretrained pieces (ContentVec; NSF-HiFiGAN ckpt URL
#    optional — without it, the vocoder falls back to a placeholder)
uv run python scripts/download_pretrained.py

# 2. Preprocess a dataset (drops raw wavs under dataset/raw/{style}/)
uv run python scripts/preprocess.py data=opensinger

# 3. Train (resolves configs/train/<profile>.yaml)
uv run python scripts/train.py train=smoke data=opensinger

# 4. Convert one vocal
uv run python scripts/infer.py input_wav=vocal.wav output_wav=vox.wav \
    +ckpt_path=checkpoints/last.pt

# 5. Launch the GUI
uv run python scripts/ui.py +ckpt_path=checkpoints/last.pt

# 6. Benchmark against a val split
uv run python scripts/benchmark.py data=opensinger +ckpt_path=checkpoints/last.pt
```

All scripts use Hydra for config resolution; `+key=value` adds new keys,
`key=value` overrides existing ones.

## Architecture (at a glance)

```
WAV → [ContentVec, F0, UV, Loudness] → Aggregator (GST + FiLM)
    → WhisperAwareConditioning → Flow Matching decoder (Euler K=4)
                                  (or DDIM K=50 for ablation)
    → NSF-HiFiGAN (F0-conditioned) → WAV
```

Inference uses EMA-averaged weights; training supports bf16 / fp16 / fp32
mixed precision. Full design notes:
[docs/architecture.md](docs/architecture.md).

## How VOX compares to related open-source SVC

| | VOX | [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc) | [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | [DiffSVC](https://github.com/prophesier/diff-svc) | [Seed-VC](https://github.com/Plachtaa/seed-vc) |
|---|---|---|---|---|---|
| Decoder | Flow Matching (default) / DDIM | VITS | VITS + retrieval | DDPM | Diffusion (zero-shot) |
| Few-step inference | Euler K=4 | — | — | DDIM K=50+ | Distilled |
| Style blending | SLERP barycentric, 3 styles | None | None | None | Reference-based |
| Whisper-aware branch | Yes (residual) | No | No | No | No |
| Language-agnostic content | ContentVec layer 12 | ContentVec | HuBERT + retrieval | ContentVec | SSL |
| Built-in eval (MCD/F0/UV/SS) | Yes | No | No | No | No |
| GUI | gradio | gradio (community) | gradio | None | None |
| Zero-shot | No (fine-tune per target) | No | Partial | No | **Yes** |
| Documentation | 4 design docs + tests | Sparse | Sparse | Sparse | Reasonable |

VOX is **not** a zero-shot system — it aims at the niche of "I record my
own ~5h of singing, train, then have a permanent personal model with
GUI + objective metrics + documentation". Seed-VC and similar 2024-2025
zero-shot work address a different niche.

## Documentation

| File | Topic |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Module map, tensor conventions, design decisions |
| [docs/data_pipeline.md](docs/data_pipeline.md) | Feature extractors, chunking, QA, Dataset, preprocess CLI |
| [docs/training.md](docs/training.md) | Trainer, losses, optimizer, logging, checkpoints |
| [docs/inference.md](docs/inference.md) | InferencePipeline, Auto-Tune, Style blend, CLI / GUI |
| [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) | What this engine may and may not be used for |
| [MODEL_CARD.md](MODEL_CARD.md) | Capabilities, limitations, training data, biases |

## References

VOX integrates ideas from the following work. We do **not** claim novelty
on these techniques; the contribution is the integration, evaluation
harness, GUI, and documentation.

- ContentVec — Qian et al., *ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers* (ICML 2022).
- Flow Matching — Lipman et al., *Flow Matching for Generative Modeling* (2022); Liu et al., *Rectified Flow* (2022).
- Diffusion baseline — DiffSinger (Liu et al., 2022) and DiffSVC.
- NSF-HiFiGAN — Source-Filter HiFi-GAN, Yoneyama et al. (2023).
- F0 extraction — RMVPE (2023); torchcrepe (CREPE 2018) as a fallback.
- v-prediction — Salimans & Ho, *Progressive Distillation* (2022).
- EMA in generative training — Karras et al.; Stable Diffusion 2 (decay 0.9999).
- Audio domain Flow Matching — FlashAudio (2024), RFWave (ICLR 2025) — informed our scheduler choice; see [docs/architecture.md](docs/architecture.md) §2 for honest framing of the limits.
- SVC competitors compared in this README — see links in the comparison table above.

## Tests

```bash
uv run --no-sync pytest tests/ -m "not slow"   # ~15s, 161 tests
uv run --no-sync pytest tests/                  # + 3 slow tests (~120s)
```

Test fixtures are deliberately tiny (B=2, T=20-40, hidden=32) so the full
fast suite runs in single-digit seconds on CPU. CUDA-specific tests
(bf16 autocast on GPU) skip automatically when CUDA is unavailable.

## Funding / origin

VOX is being submitted as part of the Ryukoku Challenge 2026 grant
application. The application describes Phase B in detail (recording,
production training, song release, technical blog). Phase A (this
repository) was completed before submission to demonstrate feasibility.

## License

Apache-2.0. See [LICENSE](LICENSE). For voice-cloning ethics and
deepfake risk, see [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md).
