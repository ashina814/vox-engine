"""Automatic chunk-level QA for the data pipeline.

Each check returns ``(passed, metric)``. ChunkQA aggregates them per the
thresholds in ``QAConfig`` and produces a ``QAResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class QAConfig:
    f0_jump_threshold: float = 0.05  # max ratio of >1-semitone jumps
    f0_min_hz: float = 50.0
    f0_max_hz: float = 800.0
    energy_std_min: float = 0.02
    silence_ratio_max: float = 0.5
    min_duration_s: float = 2.0
    max_duration_s: float = 12.0
    clip_ratio_max: float = 0.001


@dataclass
class QAResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


def _as_np(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def check_f0_continuity(f0: Tensor | np.ndarray, max_jump_ratio: float = 0.05) -> tuple[bool, float]:
    """Voiced-frame F0 must not jump by >= 1 semitone too often.

    Ratio = (#frames with |Δlog2(f0)| >= 1/12) / (#voiced transitions).
    """
    f = _as_np(f0).astype(np.float64)
    voiced = f > 0
    # transitions where both endpoints are voiced
    pair = voiced[:-1] & voiced[1:]
    if pair.sum() == 0:
        return True, 0.0
    delta = np.abs(np.log2(f[1:][pair] / f[:-1][pair]))
    jump_ratio = float((delta >= 1.0 / 12.0).mean())
    return jump_ratio < max_jump_ratio, jump_ratio


def check_f0_range(
    f0: Tensor | np.ndarray,
    uv: Tensor | np.ndarray,
    fmin: float = 50.0,
    fmax: float = 800.0,
) -> tuple[bool, float]:
    """Mean F0 over voiced frames must lie in [fmin, fmax]."""
    f = _as_np(f0)
    voiced = _as_np(uv).astype(bool)
    if voiced.sum() == 0:
        return False, 0.0
    mean_f0 = float(f[voiced].mean())
    return (fmin <= mean_f0 <= fmax), mean_f0


def check_energy_dynamic_range(
    loudness: Tensor | np.ndarray, min_std: float = 0.02
) -> tuple[bool, float]:
    """Loudness std must exceed min_std (rejects flat / dead signals)."""
    std = float(_as_np(loudness).std())
    return std > min_std, std


def check_silence_ratio(uv: Tensor | np.ndarray, max_ratio: float = 0.5) -> tuple[bool, float]:
    """Unvoiced frame ratio must be below max_ratio."""
    u = _as_np(uv).astype(bool)
    if u.size == 0:
        return False, 1.0
    ratio = float((~u).mean())
    return ratio < max_ratio, ratio


def check_duration(duration_s: float, min_s: float = 2.0, max_s: float = 12.0) -> tuple[bool, float]:
    return (min_s <= duration_s <= max_s), float(duration_s)


def check_clip_ratio(wav: Tensor | np.ndarray, max_ratio: float = 0.001) -> tuple[bool, float]:
    """Fraction of |x| > 0.99 must be below max_ratio."""
    w = _as_np(wav)
    if w.size == 0:
        return False, 1.0
    ratio = float((np.abs(w) > 0.99).mean())
    return ratio < max_ratio, ratio


@dataclass
class ChunkFeatures:
    """Bundle passed to ChunkQA.__call__."""

    wav: Tensor
    sr: int
    f0: Tensor
    uv: Tensor
    loudness: Tensor

    @property
    def duration_s(self) -> float:
        return self.wav.shape[-1] / self.sr


class ChunkQA:
    def __init__(self, config: QAConfig | None = None) -> None:
        self.cfg = config or QAConfig()

    def __call__(self, c: ChunkFeatures) -> QAResult:
        cfg = self.cfg
        checks: dict[str, bool] = {}
        metrics: dict[str, float] = {}

        passed, m = check_duration(c.duration_s, cfg.min_duration_s, cfg.max_duration_s)
        checks["duration"] = passed
        metrics["duration_s"] = m

        passed, m = check_clip_ratio(c.wav, cfg.clip_ratio_max)
        checks["clip_ratio"] = passed
        metrics["clip_ratio"] = m

        passed, m = check_f0_continuity(c.f0, cfg.f0_jump_threshold)
        checks["f0_continuity"] = passed
        metrics["f0_jump_ratio"] = m

        passed, m = check_f0_range(c.f0, c.uv, cfg.f0_min_hz, cfg.f0_max_hz)
        checks["f0_range"] = passed
        metrics["f0_mean_hz"] = m

        passed, m = check_energy_dynamic_range(c.loudness, cfg.energy_std_min)
        checks["energy_dynamic_range"] = passed
        metrics["loudness_std"] = m

        passed, m = check_silence_ratio(c.uv, cfg.silence_ratio_max)
        checks["silence_ratio"] = passed
        metrics["silence_ratio"] = m

        all_passed = all(checks.values())
        reason = None if all_passed else ",".join(k for k, v in checks.items() if not v)
        return QAResult(passed=all_passed, checks=checks, metrics=metrics, reason=reason)
