# Data pipeline

## Directory layout

```
dataset/
├── raw/{style}/{take}.wav               # input
├── processed/
│   ├── mel/{style}/{chunk_id}.npy       # (128, T)
│   ├── f0/{style}/{chunk_id}.npy        # (T,)
│   ├── uv/{style}/{chunk_id}.npy        # (T,) bool
│   ├── content/{style}/{chunk_id}.npy   # (768, T) — optional
│   └── loudness/{style}/{chunk_id}.npy  # (T,)
├── quarantine/{chunk_id}.json           # QA rejections + reasons
└── index.parquet                        # one row per accepted chunk
```

`index.parquet` columns:

| column | type |
|---|---|
| `chunk_id` | str |
| `style` / `style_id` | str / int |
| `split` | `"train"` / `"val"` |
| `duration_s` | float |
| `source_wav` | str (absolute path to the raw wav) |
| `mel_path`, `f0_path`, `uv_path`, `content_path`, `loudness_path` | str |

## Feature extractors (`src/vox/data/features/`)

| Module | Class / Fn | Notes |
|---|---|---|
| `mel.py` | `MelExtractor` | 44.1 kHz, n_mels=128, n_fft=2048, hop=512, win=2048, `center=True` |
| `f0.py` | `F0Extractor(backend="torchcrepe")` | RMVPE backend stubbed for Phase B |
| `uv.py` | `compute_uv(f0, energy)` | voiced = f0 > 0 ∧ energy > threshold |
| `loudness.py` | `a_weighted_rms` | IEC 61672 A-weight via bilinear transform, frame-by-frame |
| `content.py` | `ContentVecExtractor` | HuBERT layer 12, frozen, internal 16 kHz |

All extractors are time-aligned to the same `T = T_wav // hop + 1` grid;
F0 and content are post-resampled to that count via linear interpolation.

## Chunking (`chunking.py`)

`chunk_wav(wav, sr, min_s, max_s, silence_db, min_silence_s)`:

1. `librosa.effects.split` finds non-silent intervals.
2. Greedy merge until reaching `max_s`.
3. Hard-split anything longer than `max_s`.
4. Drop chunks under `min_s`.

Returns a list of `Chunk` dataclasses with sample-level provenance
(`start_sample`, `end_sample`) — useful for tracing a feature back to the
source wav at evaluation time.

## QA (`qa.py`)

`ChunkQA(QAConfig)` runs six per-chunk checks:

| Check | Function | Default threshold |
|---|---|---|
| Duration | `check_duration` | 2.0 ≤ s ≤ 12.0 |
| Clip ratio | `check_clip_ratio` | < 0.1% samples with |x| > 0.99 |
| F0 continuity | `check_f0_continuity` | < 5% of voiced transitions jump ≥ 1 semitone |
| F0 range | `check_f0_range` | mean voiced F0 ∈ [50, 800] Hz |
| Energy dynamics | `check_energy_dynamic_range` | loudness std > 0.02 |
| Silence ratio | `check_silence_ratio` | unvoiced fraction < 50% |

Failures are written as `quarantine/{chunk_id}.json` with the list of failed
checks and the offending metric values.

## Dataset & collation

`VoxDataset(index_path, split="train")` lazy-loads `.npy` features and
returns a dict per chunk. `collate_fn(batch)` pads variable-length features
to the batch max along the time axis and emits a `mask` tensor (1 on valid
frames).

## CLI

```bash
# Default config (configs/base.yaml) — opensinger profile
uv run python scripts/preprocess.py data=opensinger

# Skip ContentVec extraction for a smoke run
uv run python scripts/preprocess.py data=synthetic +preprocess.extract_content=false
```

The CLI delegates to `run_preprocessing(cfg, extractors)` which is **the same
function used in unit tests with mocked extractors** — keeping production
and test paths identical.
