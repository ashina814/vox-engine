"""Gradio frontend for the VOX inference pipeline.

The app is intentionally a thin shell: every parameter maps 1:1 to an
``InferenceRequest`` field, so any improvement that lands in the pipeline is
immediately exposed here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vox.inference.pipeline import InferencePipeline, InferenceRequest


def build_app(pipeline: InferencePipeline, n_styles: int = 3):
    """Return a ``gr.Blocks`` UI bound to ``pipeline``.

    Gradio is imported lazily so unit tests that only assert the wiring do not
    need the optional dependency installed in CI.
    """
    import gradio as gr

    style_labels = ["Normal", "Whisper", "Power", "Style 4", "Style 5"][: n_styles]

    def _normalize(weights: list[float]) -> tuple[float, ...]:
        weights = [max(0.0, float(w)) for w in weights]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] + [0.0] * (len(weights) - 1)
            total = 1.0
        return tuple(w / total for w in weights)

    def run(
        input_path: str | None,
        target_key: str,
        autotune_strength: float,
        num_steps: int,
        *style_sliders: float,
    ):
        if not input_path:
            return None, "No input audio provided."
        weights = _normalize(list(style_sliders))
        key = None if target_key in (None, "None", "") else target_key
        req = InferenceRequest(
            input_wav=Path(input_path),
            target_key=key,
            autotune_strength=float(autotune_strength),
            num_diffusion_steps=int(num_steps),
            style_weights=weights,
        )
        res = pipeline(req)
        info = (
            f"styles={[round(w, 3) for w in weights]}  "
            f"rtf={res.metadata.get('rtf', float('nan')):.2f}  "
            f"steps={res.metadata.get('num_diffusion_steps')}"
        )
        return (res.sr, res.output_wav.astype(np.float32)), info

    with gr.Blocks(title="VOX") as app:
        gr.Markdown("# VOX — Singing Voice Conversion")
        with gr.Row():
            input_audio = gr.Audio(label="Input vocal", type="filepath")
            output_audio = gr.Audio(label="VOX output", type="numpy")
        with gr.Row():
            target_key = gr.Dropdown(
                ["C", "D", "E", "F", "G", "A", "B", "None"],
                value="C",
                label="Target key (None = no Auto-Tune)",
            )
            autotune = gr.Slider(0.0, 1.0, value=0.8, label="Auto-Tune strength")
            num_steps = gr.Slider(1, 100, value=50, step=1, label="Diffusion steps")
        style_sliders = []
        with gr.Row():
            for i, label in enumerate(style_labels):
                init = 1.0 if i == 0 else 0.0
                style_sliders.append(gr.Slider(0.0, 1.0, value=init, label=label))
        run_btn = gr.Button("Convert")
        meta = gr.Textbox(label="Info", interactive=False)
        run_btn.click(
            fn=run,
            inputs=[input_audio, target_key, autotune, num_steps, *style_sliders],
            outputs=[output_audio, meta],
        )
    return app


def launch(pipeline: InferencePipeline, n_styles: int = 3, **launch_kwargs) -> None:
    """Convenience wrapper that builds and launches the app in one call."""
    build_app(pipeline, n_styles=n_styles).launch(**launch_kwargs)
