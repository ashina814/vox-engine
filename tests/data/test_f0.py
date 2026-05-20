"""F0 extractor tests – CPU only, no model download needed for basic shape test."""
import pytest
import torch

from vox.data.features.f0 import F0Extractor
from vox.data.features.mel import MelExtractor


@pytest.fixture
def mel_frame_count(sine_wav):
    ext = MelExtractor()
    mel = ext(sine_wav)
    return mel.shape[-1]


def test_f0_requires_1d():
    ext = F0Extractor()
    with pytest.raises(ValueError):
        ext(torch.randn(2, 1000))


@pytest.mark.slow
def test_f0_sine_accuracy(sine_wav, mel_frame_count):
    """440 Hz sine → F0 should be close to 440 Hz on voiced frames."""
    ext = F0Extractor(backend="torchcrepe")
    f0, confidence = ext(sine_wav)

    assert f0.shape == (mel_frame_count,), f"F0 shape mismatch: {f0.shape}"
    assert confidence.shape == (mel_frame_count,)

    voiced_f0 = f0[f0 > 0]
    assert len(voiced_f0) > 0, "No voiced frames detected"

    median_f0 = voiced_f0.median().item()
    assert abs(median_f0 - 440.0) < 20.0, f"F0 median {median_f0:.1f} Hz, expected ~440 Hz"
