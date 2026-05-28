# VOX — Model Card

Model cards describe a model's capabilities, limitations, training data,
and intended use, following the framework of Mitchell et al. (2019).
This card covers the **VOX engine architecture itself**. When you train a
specific voice with VOX (Phase B), generate a derived model card for
that voice using this file as a template.

---

## 1. Model overview

| Field | Value |
|---|---|
| Model name | VOX Singing Voice Conversion engine |
| Version | Phase A complete (2026-05-20) |
| Repository | https://github.com/ashina814/vox-engine |
| License | Apache-2.0 |
| Acceptable Use | [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) |
| Task | Singing Voice Conversion (SVC) — re-synthesise a vocal in another singer's voice / style |
| Architecture | ContentVec encoder (frozen) + Condition Aggregator + Diffusion / Flow Matching decoder + NSF-HiFiGAN vocoder |
| Frameworks | PyTorch 2.x, Python 3.11 |

## 2. Intended use

**Primary intended uses**:

- Self-voice singing synthesis: a user trains the model on their own
  voice and uses it to sing in a different style or after losing the
  ability to sing.
- Singing voice conversion with explicit consent from the source singer.
- Research and education on diffusion-based vocal synthesis.
- Inspiration / template for similar open-source SVC projects.

**Out-of-scope uses** — see [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) §2 for
the full list. Briefly:

- Cloning a real person's voice without consent.
- Generating non-consensual audio of any identifiable person.
- Bypassing voice authentication.
- Anything involving a minor's voice.

## 3. Architecture details

| Component | Role | Source |
|---|---|---|
| ContentVec-768 (frozen) | Language-agnostic content features | `lengyue233/content-vec-best` (HuggingFace, HuBERT-based) |
| F0 extractor | Pitch contour | torchcrepe (default), RMVPE backend stub for Phase B |
| Loudness extractor | A-weighted RMS | scipy (IEC 61672 bilinear transform) |
| Condition Aggregator | Combine content + F0 + UV + loudness + style | `src/vox/models/conditioning/aggregator.py` |
| Whisper Branch | Style-aware residual for whisper vocals | `src/vox/models/conditioning/whisper_branch.py` |
| Diffusion / Flow Matching decoder | Mel-spectrogram generator | `src/vox/models/diffusion/` |
| NSF-HiFiGAN vocoder | Mel → waveform, F0-conditioned | `src/vox/models/vocoder/nsf_hifigan.py` |

**Inference**: Flow Matching (default) with Euler K=4 steps, or DDIM K=50
for the legacy diffusion path. EMA-averaged weights are used at inference.

## 4. Training data

The Phase A reference implementation **ships no trained checkpoint**. The
weights you produce depend entirely on the data you provide.

**Phase B recommended dataset** (for self-voice):

- ~5.5 hours of own singing across three styles (normal / whisper / power)
- Recorded in a low-noise environment (≤ −55 dBFS noise floor recommended)
- 44.1 kHz, ≥ 24-bit, monophonic
- Per-style consistency: same microphone, same room, same distance

**Publicly trainable datasets** (for research / baseline experiments):

- OpenSinger (Mandarin, ~50h multi-speaker) — for SVC architecture
  validation
- JVS-MuSiC (Japanese, ~2h × 100 speakers) — for Japanese-language
  characteristics

If you train on another person's voice or on copyrighted material, you
must comply with the consent and licensing requirements in
[ACCEPTABLE_USE.md](ACCEPTABLE_USE.md) §2.

## 5. Evaluation

Built-in objective metrics:

| Metric | Module | Phase B target |
|---|---|---|
| MCD (mel-cepstral distortion, dB) | `vox.evaluation.mcd` | < 6.0 |
| log F0 RMSE | `vox.evaluation.f0_rmse` | < 0.15 |
| UV error rate | `vox.evaluation.uv_error` | < 5% |
| Style separability (classifier acc.) | `vox.evaluation.style_separability` | > 80% |
| Real Time Factor | `vox.evaluation.benchmark` | < 0.5 on consumer GPU |

The Phase A implementation **has not been validated against these
targets on real singing data**. Validation is part of Phase B.

## 6. Limitations

- **No pretrained checkpoint is distributed** in Phase A. Users must
  train on their own data.
- **Fine-tune-based, not zero-shot.** Requires per-target training. Newer
  zero-shot approaches (Seed-VC, YingMusic-SVC, 2024-2025) work from a
  short reference clip and may suit different use cases better.
- **Small-data regime untested.** Reasonable behaviour at ≥ 5h per
  speaker is the design assumption; below that, results are unknown.
- **English vocals untested.** ContentVec is language-agnostic in
  principle but Phase A development was in Japanese context only.
- **Whisper-branch effectiveness unproven.** The residual is plausibly
  motivated but its quantitative benefit over plain style embeddings
  has not been measured.
- **Vocoder is a placeholder by default.** Until a real NSF-HiFiGAN
  checkpoint is downloaded via `scripts/download_pretrained.py`, audio
  output is unusable for human listening (a smoke generator).
- **No safety classifier / watermark / consent verification** is built
  in. Misuse protection relies on this Acceptable Use Policy.

## 7. Known biases and risks

- **Voice identity bias**: A model trained on a narrow voice produces
  outputs that sound like that voice regardless of input. This is the
  intended behaviour but means the model **cannot impartially represent
  multiple identities** from a single checkpoint.
- **Stylistic bias**: Whisper / power / normal categories are coarse.
  Subtler emotion (e.g. quiet sadness vs. tender) is not directly
  controllable.
- **Cultural bias**: Reference design assumed Japanese / East-Asian
  popular-music vocal aesthetics. Performance on operatic, traditional,
  or non-melodic vocal styles is unknown.
- **Misuse risk**: As with any voice synthesis system, the model can be
  trained on unconsented audio. See ACCEPTABLE_USE.md for mitigations
  the authors require of users.

## 8. Environmental / compute cost

A Phase B production run (100k step, RTX 4090) is estimated at
~30-50 kWh including dataset preprocessing. Inference on consumer GPU is
~0.1-0.5 kWh per hour of generated audio. Use of cloud GPU (RunPod
spot pricing) is the assumed deployment mode for budget reasons.

## 9. Contact

Issues and questions: https://github.com/ashina814/vox-engine/issues

---

*Document Version: 1.0 / Date: 2026-05-20*
*Template format follows Mitchell et al., "Model Cards for Model Reporting", FAT* 2019.*
