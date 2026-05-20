import numpy as np
import pytest

from vox.inference.autotune import (
    cents_to_hz,
    get_scale,
    hz_to_cents,
    preserve_vibrato,
    snap_to_scale,
)


def test_get_scale_c_major():
    assert get_scale("C", "major") == [0, 2, 4, 5, 7, 9, 11]


def test_get_scale_a_minor_equals_c_major():
    assert sorted(get_scale("A", "minor")) == sorted(get_scale("C", "major"))


def test_get_scale_unknown_key():
    with pytest.raises(ValueError):
        get_scale("Q")


def test_hz_to_cents_a440_anchor():
    cents = hz_to_cents(np.array([440.0]))
    assert abs(cents[0] - 5700.0) < 1e-3


def test_hz_to_cents_zero_unvoiced():
    cents = hz_to_cents(np.array([0.0, 440.0, 0.0]))
    assert cents[0] == 0.0
    assert cents[2] == 0.0


def test_cents_roundtrip():
    hz = np.array([110.0, 220.0, 440.0, 880.0], dtype=np.float32)
    back = cents_to_hz(hz_to_cents(hz))
    assert np.allclose(back, hz, atol=1e-2)


def test_snap_440_stays_440():
    f0 = np.full(200, 440.0, dtype=np.float32)
    out = snap_to_scale(f0, get_scale("C"), strength=1.0, smooth_ms=0.0)
    assert np.allclose(out, 440.0, atol=1.0)


def test_snap_450_pulls_toward_440():
    """A 450 Hz line (≈39 cents sharp of A) should pull toward 440 with strength=1."""
    f0 = np.full(200, 450.0, dtype=np.float32)
    out = snap_to_scale(f0, get_scale("C"), strength=1.0, smooth_ms=0.0)
    # Center frames (post moving-avg) should be much closer to 440 than 450.
    mid = out[80:120]
    assert np.abs(mid - 440.0).mean() < np.abs(mid - 450.0).mean()


def test_snap_preserves_unvoiced():
    f0 = np.array([0.0, 0.0, 440.0, 440.0, 0.0], dtype=np.float32)
    out = snap_to_scale(f0, get_scale("C"), strength=1.0, smooth_ms=0.0)
    assert out[0] == 0.0 and out[1] == 0.0 and out[-1] == 0.0


def test_snap_strength_zero_is_identity():
    f0 = np.linspace(200.0, 600.0, 100, dtype=np.float32)
    out = snap_to_scale(f0, get_scale("C"), strength=0.0)
    assert np.allclose(out, f0, atol=1.0)


def test_vibrato_split_recomposes_to_original():
    """melody + vibrato (in cents) should reconstruct hz_to_cents(f0)."""
    rng = np.random.default_rng(0)
    base_cents = 5700.0 + 50.0 * np.sin(np.linspace(0, 4 * np.pi, 400))
    f0 = cents_to_hz(base_cents)
    f0 = f0 + rng.normal(0, 0.5, size=f0.shape).astype(np.float32)  # tiny noise
    melody, vibrato = preserve_vibrato(f0)
    voiced = f0 > 0
    recon = melody + vibrato
    # On voiced frames, melody + vibrato == hz_to_cents(f0) (exactly, by construction).
    assert np.allclose(recon[voiced], hz_to_cents(f0)[voiced], atol=1e-3)


def test_vibrato_melody_is_smoother_than_full_signal():
    # Frame rate = 44100/512 ≈ 86 Hz; 120π over 400 frames → ~12.9 Hz vibrato,
    # clearly above the 6 Hz LPF cutoff so the filter must attenuate it.
    cents = 5700.0 + 200.0 * np.sin(np.linspace(0, 120 * np.pi, 400))
    f0 = cents_to_hz(cents)
    melody, _ = preserve_vibrato(f0, cutoff_hz=6.0)
    assert melody.std() < hz_to_cents(f0).std() * 0.5
