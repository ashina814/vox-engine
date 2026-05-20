"""Smoke test: app builds and the wired callback runs end-to-end."""
import numpy as np
import soundfile as sf
import torch
import pytest

from vox.inference.pipeline import InferencePipeline
from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig


pytest.importorskip("gradio", reason="gradio not installed in this environment")


def _tiny_pipeline():
    cfg = VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=100,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )

    def f0_fn(wav):
        T = wav.shape[-1] // 512 + 1
        return torch.full((T,), 440.0), torch.full((T,), 0.9)

    def content_fn(wav, sr):
        T = wav.shape[-1] // 512 + 1
        return torch.randn(768, T)

    return InferencePipeline(VoxModel(cfg).eval(), f0_fn=f0_fn, content_fn=content_fn)


def test_build_app_returns_blocks():
    from vox.ui.gradio_app import build_app

    app = build_app(_tiny_pipeline(), n_styles=3)
    # Gradio Blocks expose a 'blocks' attribute holding the component tree.
    assert hasattr(app, "blocks")


def test_app_callback_runs(tmp_path):
    """Invoke the same `run` callback the UI button is bound to."""
    from vox.ui.gradio_app import build_app

    pipeline = _tiny_pipeline()
    app = build_app(pipeline, n_styles=3)

    wav = (0.1 * np.random.RandomState(0).randn(int(0.3 * 44100))).astype(np.float32)
    path = tmp_path / "in.wav"
    sf.write(path, wav, 44100)

    # Pull the click handler off the first registered button.
    fn = next(
        d.fn for d in app.fns.values()
        if d.fn is not None and getattr(d.fn, "__name__", "") == "run"
    )
    out_audio, info = fn(str(path), "None", 0.0, 3, 1.0, 0.0, 0.0)
    sr, audio = out_audio
    assert sr == 44_100
    assert audio.dtype == np.float32
    assert audio.shape[0] > 0
    assert "rtf=" in info


def test_app_callback_no_input_returns_none():
    from vox.ui.gradio_app import build_app

    app = build_app(_tiny_pipeline(), n_styles=3)
    fn = next(
        d.fn for d in app.fns.values()
        if d.fn is not None and getattr(d.fn, "__name__", "") == "run"
    )
    out_audio, info = fn(None, "C", 0.8, 3, 1.0, 0.0, 0.0)
    assert out_audio is None
    assert "No input" in info
