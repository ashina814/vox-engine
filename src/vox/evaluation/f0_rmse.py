"""F0 RMSE in log space, restricted to mutually-voiced frames."""

from __future__ import annotations

import numpy as np


def f0_rmse(
    f0_ref: np.ndarray,
    f0_pred: np.ndarray,
    uv: np.ndarray | None = None,
    log: bool = True,
) -> float:
    """Frame-wise F0 RMSE.

    Args:
        f0_ref / f0_pred: (T,) Hz. 0 marks unvoiced and is excluded.
        uv: optional mutual-voiced mask. When omitted, voiced = (ref > 0) & (pred > 0).
        log: when True, RMSE is computed in log-Hz space (the default and the
             value used by the design spec target).

    Returns:
        Scalar RMSE. Returns 0.0 when there are no comparable frames.
    """
    ref = np.asarray(f0_ref, dtype=np.float64)
    pred = np.asarray(f0_pred, dtype=np.float64)
    if ref.shape != pred.shape:
        raise ValueError(f"f0_ref shape {ref.shape} != f0_pred shape {pred.shape}")

    if uv is None:
        mask = (ref > 0) & (pred > 0)
    else:
        mask = np.asarray(uv, dtype=bool) & (ref > 0) & (pred > 0)

    if mask.sum() == 0:
        return 0.0

    if log:
        diff = np.log(ref[mask]) - np.log(pred[mask])
    else:
        diff = ref[mask] - pred[mask]
    return float(np.sqrt(np.mean(diff**2)))
