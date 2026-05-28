"""End-to-end smoke test for the inference pipeline with mocked extractors."""

import numpy as np
import pytest
import torch

from vox.inference.pipeline import InferencePipeline, InferenceRequest
from vox.models.conditioning.aggregator import AggregatorConfig
from vox.models.diffusion.decoder import DecoderConfig
from vox.models.vox_model import VoxModel, VoxModelConfig


def _tiny_model() -> VoxModel:
    cfg = VoxModelConfig(
        hidden=32,
        n_styles=3,
        diffusion_steps=100,
        aggregator=AggregatorConfig(hidden=32, n_styles=3, gst_num_tokens=4, gst_num_heads=4),
        decoder=DecoderConfig(hidden=32, cond_dim=32, num_blocks=2, time_dim=32),
    )
    return VoxModel(cfg).eval()


def _fake_f0(wav: torch.Tensor):
    T = wav.shape[-1] // 512 + 1
    return torch.full((T,), 440.0), torch.full((T,), 0.9)


def _fake_content(wav: torch.Tensor, sr: int):
    T = wav.shape[-1] // 512 + 1
    return torch.randn(768, T)


@pytest.fixture
def pipeline():
    return InferencePipeline(_tiny_model(), f0_fn=_fake_f0, content_fn=_fake_content)


def test_pipeline_runs_on_array_input(pipeline):
    wav = np.random.RandomState(0).randn(int(0.5 * 44100)).astype(np.float32) * 0.1
    req = InferenceRequest(
        input_wav=wav, num_sampling_steps=3, target_key=None, style_weights=(1.0, 0.0, 0.0)
    )
    res = pipeline(req)
    assert res.output_wav.dtype == np.float32
    assert res.output_wav.shape[0] > 0
    assert np.isfinite(res.output_wav).all()
    assert res.sr == 44_100


def test_pipeline_metadata_fields(pipeline):
    wav = np.zeros(int(0.3 * 44100), dtype=np.float32)
    res = pipeline(InferenceRequest(input_wav=wav, num_sampling_steps=2, target_key=None))
    assert "elapsed_s" in res.metadata
    assert "rtf" in res.metadata
    assert res.metadata["num_sampling_steps"] == 2
    assert res.metadata["autotune_applied"] is False


def test_pipeline_autotune_flag_when_target_key_set(pipeline):
    wav = np.random.RandomState(1).randn(int(0.4 * 44100)).astype(np.float32) * 0.1
    res = pipeline(
        InferenceRequest(input_wav=wav, num_sampling_steps=2, target_key="C", autotune_strength=0.5)
    )
    assert res.metadata["autotune_applied"] is True


def test_pipeline_loads_wav_file(tmp_path, pipeline):
    import soundfile as sf

    wav = (0.1 * np.random.RandomState(2).randn(int(0.3 * 44100))).astype(np.float32)
    path = tmp_path / "in.wav"
    sf.write(path, wav, 44100)
    res = pipeline(InferenceRequest(input_wav=path, num_sampling_steps=2, target_key=None))
    assert res.output_wav.shape[0] > 0


def test_pipeline_rejects_too_many_style_weights(pipeline):
    wav = np.zeros(int(0.3 * 44100), dtype=np.float32)
    with pytest.raises(ValueError):
        pipeline(
            InferenceRequest(
                input_wav=wav,
                num_sampling_steps=2,
                style_weights=(0.25, 0.25, 0.25, 0.25),  # > n_styles=3
            )
        )


def test_pipeline_style_weights_change_output(pipeline):
    torch.manual_seed(0)
    wav = (0.1 * np.random.RandomState(3).randn(int(0.4 * 44100))).astype(np.float32)
    req_a = InferenceRequest(
        input_wav=wav, num_sampling_steps=2, target_key=None, style_weights=(1.0, 0.0, 0.0)
    )
    req_b = InferenceRequest(
        input_wav=wav, num_sampling_steps=2, target_key=None, style_weights=(0.0, 0.0, 1.0)
    )
    res_a = pipeline(req_a)
    res_b = pipeline(req_b)
    # Sanity: the two outputs should differ (style channel actually wired in).
    assert not np.allclose(res_a.output_wav, res_b.output_wav, atol=1e-4)
