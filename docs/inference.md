# Inference

## Pipeline

`vox.inference.pipeline.InferencePipeline`:

```python
pipeline = InferencePipeline(
    model=VoxModel(...),
    f0_fn=F0Extractor(...),
    content_fn=ContentVecExtractor(),
)

result = pipeline(InferenceRequest(
    input_wav=Path("vocal.wav"),
    target_key="C",
    target_mode="major",
    style_weights=(0.6, 0.0, 0.4),
    autotune_strength=0.8,
    num_sampling_steps=50,
    ref_mel=None,  # or (n_mels, T_ref) for GST
))

# result.output_wav: np.float32, sr=44_100
# result.metadata: {"elapsed_s", "rtf", "num_sampling_steps",
#                   "style_weights", "autotune_applied", "T_mel"}
```

Internally:

1. Load WAV (soundfile, mono, resample as needed).
2. Run extractors: mel, F0, loudness, UV, content.
3. Auto-Tune the F0 (skipped when `target_key=None`).
4. Resolve the style vector via SLERP barycentric on
   `model.aggregator.style_emb.weight`.
5. Align everything to `T = T_mel`.
6. `model.infer(content, f0, uv, loudness, style_id=0, style_vec=blended,
   num_steps=...)`.
7. Vocoder returns wav, pipeline wraps it in `InferenceResult`.

## Auto-Tune (`vox.inference.autotune`)

`snap_to_scale(f0, scale, strength, smooth_ms)`:

1. `preserve_vibrato(f0)` → low-pass-filtered `melody` and high-frequency
   `vibrato`, both in cents. F0=0 stays unvoiced.
2. Snap `melody` to the nearest in-scale semitone (across the local octave
   neighbourhood).
3. Blend `(1 - strength) * melody + strength * snapped`.
4. Moving average over `smooth_ms`.
5. Add `vibrato` back and convert cents → Hz.

`get_scale(key, mode)` returns the relevant semitone class list (e.g.
`get_scale("C", "major") == [0, 2, 4, 5, 7, 9, 11]`).

## Style blending (`vox.inference.style_blend`)

`slerp(v0, v1, alpha)` does spherical interpolation along the last axis with
a stable linear fallback for near-collinear inputs.

`slerp_barycentric(vectors, weights)` extends this to N vectors by folding
the heaviest pairwise SLERPs first. A one-hot weight short-circuits to the
corresponding vector for numerical exactness.

The pipeline calls this on the rows of `aggregator.style_emb.weight` and
passes the resulting `style_vec` directly to `model.infer`, bypassing the
discrete `style_emb` lookup.

## CLI

```bash
uv run python scripts/infer.py input_wav=in.wav output_wav=out.wav
uv run python scripts/infer.py input_wav=in.wav output_wav=out.wav \
    inference.target_key=D \
    inference.num_sampling_steps=30 \
    inference.style_weights="[0.6, 0.4, 0.0]"
uv run python scripts/infer.py input_wav=in.wav output_wav=out.wav \
    +ckpt_path=ckpts/step_00010000.pt
```

`inference.skip_content=true` bypasses the ContentVec download (random
content features) — useful only for wiring smoke tests; the audio output is
meaningless in that mode.

## GUI

`scripts/ui.py` launches a gradio frontend (`vox.ui.gradio_app`). Sliders
expose every `InferenceRequest` field: target key dropdown,
`autotune_strength`, `num_sampling_steps`, and one slider per style class
(weights are normalised at submit time).
