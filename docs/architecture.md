# Architecture

VOX is a **Singing Voice Conversion (SVC)** engine designed to be extensible
into full SVS (score → song) later. The default decoder is trained with
**Flow Matching** (Lipman 2022 / Liu 2022) and sampled with **4 Euler steps**;
a legacy diffusion path (v-prediction + cosine + DDIM K=50) is preserved
behind `schedule_type="diffusion"` for ablation. The decoder is conditioned
on self-supervised content features, F0, voicing, loudness, and a learnable
style embedding.

> 🚧 **Phase A status**: this repository ships **no trained checkpoint**
> and the NSF-HiFiGAN vocoder defaults to a placeholder generator. The
> entire engine wires together end-to-end (139 tests + 27 CUDA / slow
> tests pass), but generated audio is **structural smoke only** until
> Phase B trains an acoustic model on real singing and a real vocoder
> checkpoint is loaded. See [README §Status](../README.md#status) and
> [MODEL_CARD.md](../MODEL_CARD.md) for what this means in practice.

## Data flow

```
Input WAV
  │
  ├──► ContentVec-768 (frozen)        ──► content (B, 768, T)
  ├──► F0 extractor (torchcrepe / RMVPE) ──► f0 (B, T), uv (B, T)
  └──► A-weighted RMS                  ──► loudness (B, T)
                                                   │
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │ Condition Aggregator        │
                                    │   · content_proj            │
                                    │   · f0_proj / loudness_proj │
                                    │   · uv_gate (sigmoid)       │
                                    │   · style_emb + speaker_emb │
                                    │   · GST (optional ref_mel)  │
                                    │   · FiLM modulation         │
                                    └──────────────┬──────────────┘
                                                   │
                                  ┌── WhisperAwareConditioning ──┐
                                  │  (style_id==WHISPER &        │
                                  │   (1-uv) gated residual)     │
                                  └──────────────┬───────────────┘
                                                 │  cond (B, hidden, T)
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ Diffusion Decoder (U-Net1D) │
                                  │   · v-prediction            │
                                  │   · cosine schedule         │
                                  │   · DDIM K=50 sampling      │
                                  └──────────────┬─────────────┘
                                                 │  mel (B, 128, T)
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ NSF-HiFiGAN wrapper         │
                                  │   (F0-conditioned)          │
                                  └──────────────┬─────────────┘
                                                 ▼
                                          Output WAV (44.1 kHz)
```

## Tensor conventions

All frame-aligned tensors share **T = T_wav // hop + 1** (`hop=512`), the same
count that `torchaudio.transforms.MelSpectrogram(center=True)` produces.

| Tensor | Shape | dtype |
|---|---|---|
| `wav` | `(B, T_wav)` | float32 |
| `mel` | `(B, 128, T)` | float32, normalised to [0, 1] |
| `f0` | `(B, T)` | float32 Hz, 0 on unvoiced |
| `uv` | `(B, T)` | bool / float32 |
| `loudness` | `(B, T)` | float32 |
| `content` | `(B, 768, T)` | float32 (resampled from ContentVec hop) |
| `style_id` | `(B,)` | long |
| `style_vec` | `(B, hidden)` | float32 (optional override of style_emb) |
| `mask` | `(B, T)` | bool (1 = valid frame) |

## Module map

```
src/vox/
├── data/                            # § data_pipeline.md
│   ├── features/{mel,f0,uv,loudness,content}.py
│   ├── chunking.py                  # silence-aware 5-10 s splits
│   ├── qa.py                        # ChunkQA + per-check fns
│   ├── dataset.py                   # VoxDataset + collate_fn
│   └── preprocessing.py             # run_preprocessing(extractors injected)
├── models/                          # § architecture.md (this file)
│   ├── conditioning/
│   │   ├── aggregator.py            # ConditionAggregator + FiLM
│   │   ├── gst.py                   # GlobalStyleTokens + ReferenceEncoder
│   │   └── whisper_branch.py        # WhisperAwareConditioning
│   ├── diffusion/
│   │   ├── decoder.py               # U-Net1D, ResBlock1D, sinusoidal t-emb
│   │   ├── noise_schedule.py        # cosine α̅, v-prediction targets
│   │   └── sampler.py               # DDIM K=50
│   ├── vocoder/nsf_hifigan.py       # wrapper + placeholder generator
│   └── vox_model.py                 # integrated model
├── training/                        # § training.md
│   ├── losses.py                    # masked diffusion / mel L1 / f0 consist
│   ├── optim.py                     # AdamW + cosine_with_warmup
│   ├── logging.py                   # Logger Protocol + stdout/TB/wandb
│   └── trainer.py                   # VoxTrainer
├── inference/                       # § inference.md
│   ├── autotune.py                  # cents-space snap + vibrato preserve
│   ├── style_blend.py               # slerp + barycentric
│   └── pipeline.py                  # InferencePipeline
├── evaluation/                      # § (this file, "Evaluation")
│   ├── mcd.py                       # DTW + MFCC stand-in
│   ├── f0_rmse.py                   # log-Hz RMSE on voiced frames
│   ├── uv_error.py                  # frame-wise misclassification
│   ├── style_separability.py        # classifier-based accuracy
│   └── benchmark.py                 # BenchmarkRunner → JSON
└── ui/gradio_app.py                 # thin Pipeline frontend
```

## Key design decisions

### 1. SSL features in place of phoneme + aligner
ContentVec-768 (frozen) replaces the explicit phoneme path. This kills the
Singing-MFA / aligner problem outright and makes the model language-agnostic
"for free" — adding a language is purely a data question, not a model
question.

### 2. Flow Matching (default) — with diffusion as a fallback
Following Lipman et al. (2022) and Liu et al. (2022, *Rectified Flow*), the
default decoder is trained on a **linear conditional flow**:
`x_t = (1-t)·x₀ + t·noise`, target velocity `v = noise - x₀`. Inference
integrates the ODE with a small number of Euler steps (default K=4).

Honest framing: this is a *few-step* setup, not 1-step. True 1-step
inference (FlashAudio 2025) requires an additional Rectified-Flow
distillation pass that we do not perform. Flow Matching is also not yet
a clear winner over diffusion in the audio domain — see FlowSE
(Interspeech 2025) for limitation discussions. Our choice is "modern
recipe with a reasonable few-step inference budget", not "state of the
art audio synthesis".

The legacy diffusion path (v-prediction + cosine schedule + DDIM K=50) is
preserved as `schedule_type="diffusion"` for ablation and reproducibility.

### 3. EMA weights + mixed precision
Both standard in modern generative training. EMA (decay 0.999) wraps the
trainer transparently and is used at inference. `bf16` autocast halves
memory and compute on Ampere+ GPUs.

### 4. Discrete + continuous style
The pipeline holds a single learnable embedding per style id **and** a
GST-extracted reference vector. Inference can either:
- pass `style_id` → `style_emb` lookup (training default)
- pass a precomputed `style_vec` (blend several style_emb rows via SLERP +
  barycentric — see `inference/style_blend.py`)
- supply `ref_mel` → GST contribution added on top

### 5. Whisper-aware residual
A second conv path, gated by `style_id == WHISPER` × `(1 - uv)`, only fires
on whisper-style unvoiced frames. Non-whisper batches skip the branch
entirely (early-return), so the cost is paid only when needed.

### 6. Vocoder fallback
`NSFHifiGANWrapper` falls back to a tiny placeholder ConvTranspose1D when no
checkpoint is present. This keeps every test, the CLI and the GUI runnable
on machines without the multi-hundred-MB pretrained vocoder download.

## Evaluation contract (`vox.evaluation`)

| Metric | Function | Target (design spec) |
|---|---|---|
| MCD (dB) | `mel_cepstral_distortion(wav_ref, wav_pred)` | < 6.0 |
| log F0 RMSE | `f0_rmse(f0_ref, f0_pred, uv)` | < 0.15 |
| UV error | `uv_error_rate(uv_ref, uv_pred)` | < 5% |
| Style separability | `StyleSeparabilityEvaluator(...)` | > 80% |
| RTF | `BenchmarkRunner` metadata | < 0.5 (GPU) |

The `BenchmarkRunner` is render-fn agnostic — pass any callable that maps a
dataset row to `(ref_wav, pred_wav, f0_ref, f0_pred, uv_ref, uv_pred,
style_id)` and the runner produces per-sample plus aggregate metrics as JSON.
