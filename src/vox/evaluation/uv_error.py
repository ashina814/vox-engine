"""Voiced/unvoiced frame-wise classification error rate."""

from __future__ import annotations

import numpy as np


def uv_error_rate(uv_ref: np.ndarray, uv_pred: np.ndarray) -> float:
    """Fraction of frames where predicted voicing disagrees with the reference."""
    ref = np.asarray(uv_ref).astype(bool).reshape(-1)
    pred = np.asarray(uv_pred).astype(bool).reshape(-1)
    if ref.shape != pred.shape:
        raise ValueError(f"uv_ref shape {ref.shape} != uv_pred shape {pred.shape}")
    if ref.size == 0:
        return 0.0
    return float((ref != pred).mean())
