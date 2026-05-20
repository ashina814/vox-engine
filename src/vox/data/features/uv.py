"""Voiced / Unvoiced flag computation."""
from __future__ import annotations

from torch import Tensor


def compute_uv(
    f0: Tensor,
    energy: Tensor | None = None,
    f0_threshold: float = 0.0,
    energy_threshold: float = 1e-4,
) -> Tensor:
    """Compute voiced/unvoiced flag per mel frame.

    A frame is voiced when:
      - f0 > f0_threshold  (set to 0 by F0Extractor for unvoiced frames)
      - energy > energy_threshold  (optional; skipped when energy is None)

    Args:
        f0: (T_mel,) F0 in Hz, 0 for unvoiced.
        energy: (T_mel,) frame-level RMS energy (optional).
        f0_threshold: frames with f0 <= this value are unvoiced.
        energy_threshold: frames with energy <= this value are unvoiced.

    Returns:
        uv: (T_mel,) bool tensor, True = voiced.
    """
    voiced = f0 > f0_threshold
    if energy is not None:
        voiced = voiced & (energy > energy_threshold)
    return voiced
