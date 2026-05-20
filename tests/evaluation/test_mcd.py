import numpy as np
import pytest

from vox.evaluation.mcd import mel_cepstral_distortion


@pytest.fixture
def sr():
    return 44_100


def _tone(sr, dur=1.0, freq=440.0, amp=0.3):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mcd_identical_waveforms_is_near_zero(sr):
    wav = _tone(sr, dur=1.0)
    assert mel_cepstral_distortion(wav, wav, sr=sr) < 1e-3


def test_mcd_different_waveforms_positive(sr):
    a = _tone(sr, dur=1.0, freq=440.0)
    b = _tone(sr, dur=1.0, freq=660.0)
    assert mel_cepstral_distortion(a, b, sr=sr) > 1.0


def test_mcd_noisy_vs_clean(sr):
    """A noise-corrupted version of a tone should yield a higher MCD than a clean copy."""
    rng = np.random.default_rng(0)
    clean = _tone(sr, dur=1.0)
    noisy = clean + 0.2 * rng.standard_normal(len(clean)).astype(np.float32)
    d_self = mel_cepstral_distortion(clean, clean, sr=sr)
    d_noise = mel_cepstral_distortion(clean, noisy, sr=sr)
    assert d_noise > d_self


def test_mcd_handles_length_mismatch(sr):
    a = _tone(sr, dur=1.0)
    b = _tone(sr, dur=1.4)
    val = mel_cepstral_distortion(a, b, sr=sr)
    assert np.isfinite(val) and val >= 0.0
