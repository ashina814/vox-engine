import numpy as np
import pytest

from vox.evaluation.f0_rmse import f0_rmse
from vox.evaluation.uv_error import uv_error_rate


# --- F0 RMSE --------------------------------------------------------------


def test_f0_rmse_identical_is_zero():
    f = np.array([220.0, 440.0, 880.0])
    assert f0_rmse(f, f) == 0.0


def test_f0_rmse_log_space():
    """Doubling F0 = 1 octave = ~0.693 in natural log."""
    a = np.array([100.0, 200.0])
    b = np.array([200.0, 400.0])  # consistent octave-up
    rmse = f0_rmse(a, b, log=True)
    assert abs(rmse - np.log(2.0)) < 1e-6


def test_f0_rmse_excludes_unvoiced():
    a = np.array([0.0, 220.0, 440.0, 0.0])
    b = np.array([110.0, 220.0, 440.0, 0.0])
    # Frame 0: ref unvoiced → excluded. Frame 3: both unvoiced → excluded.
    # Frames 1 and 2 match → RMSE 0.
    assert f0_rmse(a, b) == 0.0


def test_f0_rmse_returns_zero_for_no_overlap():
    a = np.zeros(10)
    b = np.array([220.0] * 10)
    assert f0_rmse(a, b) == 0.0


def test_f0_rmse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        f0_rmse(np.zeros(3), np.zeros(4))


# --- UV error -------------------------------------------------------------


def test_uv_error_zero_for_identical():
    uv = np.array([1, 0, 1, 1, 0], dtype=bool)
    assert uv_error_rate(uv, uv) == 0.0


def test_uv_error_one_for_inverted():
    uv = np.array([1, 0, 1, 1, 0], dtype=bool)
    assert uv_error_rate(uv, ~uv) == 1.0


def test_uv_error_fractional():
    a = np.array([1, 1, 1, 1], dtype=bool)
    b = np.array([1, 1, 0, 0], dtype=bool)
    assert abs(uv_error_rate(a, b) - 0.5) < 1e-9


def test_uv_error_shape_mismatch_raises():
    with pytest.raises(ValueError):
        uv_error_rate(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))
