"""Musical pitch correction (Auto-Tune) in cents space.

Pipeline:
    F0 [Hz]                       (unvoiced frames == 0 are preserved)
    → preserve_vibrato (LPF)     → melody [cents] + vibrato [cents]
    → snap melody to scale       → blended cents = (1-s)*melody + s*snapped
    → smooth via moving average  → stable cents
    → add vibrato back           → tuned cents → Hz
"""
from __future__ import annotations

from typing import Literal

import numpy as np

# Major / minor scale semitones relative to the tonic.
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]
# Semitone offset of each note from C.
_NOTE_OFFSETS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
                 "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
                 "A": 9, "A#": 10, "Bb": 10, "B": 11}

# Reference: A440 sits at MIDI 69, i.e. 5700 cents above C0 in our convention.
_A440_CENTS = 5700.0


def get_scale(key: str, mode: Literal["major", "minor"] = "major") -> list[int]:
    """Return the set of semitones (mod 12) belonging to ``key``-``mode``."""
    if key not in _NOTE_OFFSETS:
        raise ValueError(f"Unknown key: {key!r}")
    offset = _NOTE_OFFSETS[key]
    base = _MAJOR if mode == "major" else _MINOR
    return sorted((s + offset) % 12 for s in base)


def hz_to_cents(f0: np.ndarray) -> np.ndarray:
    """Hz → cents, centered such that A440 == 5700. F0=0 stays 0."""
    safe = np.where(f0 > 0, f0, 1.0)
    cents = 1200.0 * np.log2(safe / 440.0) + _A440_CENTS
    return np.where(f0 > 0, cents, 0.0)


def cents_to_hz(cents: np.ndarray, voiced: np.ndarray | None = None) -> np.ndarray:
    """Cents → Hz; frames where voiced=False are zeroed out."""
    hz = 440.0 * np.power(2.0, (cents - _A440_CENTS) / 1200.0)
    if voiced is not None:
        hz = np.where(voiced, hz, 0.0)
    return hz.astype(np.float32)


def preserve_vibrato(
    f0: np.ndarray, cutoff_hz: float = 6.0, sr_hop: int = 512, sr: int = 44_100
) -> tuple[np.ndarray, np.ndarray]:
    """Separate F0 in cents into (melody, vibrato) via a low-pass filter.

    The frame-rate sample rate is ``sr / sr_hop``. Output: both arrays are in
    cents, same length as ``f0``. Unvoiced frames (f0==0) propagate as 0 in
    melody and 0 in vibrato (so melody + vibrato == 0 there).
    """
    from scipy.signal import butter, sosfiltfilt

    voiced = f0 > 0
    cents = hz_to_cents(f0)

    if not voiced.any():
        return cents, np.zeros_like(cents)

    frame_sr = sr / sr_hop
    nyq = frame_sr * 0.5
    norm = min(0.99, cutoff_hz / nyq)
    order = 4
    sos = butter(order, norm, btype="low", output="sos")

    # Interpolate over unvoiced gaps so the filter doesn't ring on zero-edges.
    idx = np.arange(len(cents))
    melody_input = np.interp(idx, idx[voiced], cents[voiced])

    # sosfiltfilt needs len(x) > padlen (~3*order*2 = 24). Fall back to identity
    # for very short inputs so this never errors on tiny test fixtures.
    if len(melody_input) <= 3 * (2 * order + 1):
        melody = melody_input.astype(np.float64)
    else:
        melody = sosfiltfilt(sos, melody_input)
    vibrato = cents - melody

    # Zero out melody/vibrato where original was unvoiced.
    melody = np.where(voiced, melody, 0.0)
    vibrato = np.where(voiced, vibrato, 0.0)
    return melody.astype(np.float32), vibrato.astype(np.float32)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return x.astype(np.float32)
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _snap_cents_to_scale(cents: np.ndarray, scale_semitones: list[int]) -> np.ndarray:
    """Snap cents to nearest scale degree (any octave).

    For each frame we enumerate the 12-semitone class candidates within ±1
    semitone and pick the closest in cent distance.
    """
    if len(scale_semitones) == 0:
        return cents
    # All scale notes across the relevant octave range.
    semis_in_octave = np.array(scale_semitones, dtype=np.float64)
    # Convert cents to fractional semitones relative to C0.
    semi = cents / 100.0
    # For each frame, snap to the closest in-scale semitone.
    base_octave = np.floor(semi / 12.0)
    # Candidates: scale notes in this and the two adjacent octaves.
    cand = np.concatenate(
        [
            (np.expand_dims(base_octave, -1) + k) * 12.0 + semis_in_octave
            for k in (-1, 0, 1)
        ],
        axis=-1,
    )  # (T, 3*S)
    diff = np.abs(cand - semi[:, None])
    nearest_idx = diff.argmin(axis=-1)
    snapped_semis = cand[np.arange(len(semi)), nearest_idx]
    return (snapped_semis * 100.0).astype(np.float32)


def snap_to_scale(
    f0: np.ndarray,
    scale_semitones: list[int],
    strength: float = 0.8,
    smooth_ms: float = 50.0,
    sr_hop: int = 512,
    sr: int = 44_100,
    preserve_vib: bool = True,
) -> np.ndarray:
    """Snap voiced F0 frames to ``scale_semitones`` in cents space.

    Unvoiced frames (f0==0) are preserved as-is.
    """
    if strength <= 0 or len(f0) == 0:
        return f0.astype(np.float32)

    voiced = f0 > 0
    if not voiced.any():
        return f0.astype(np.float32)

    if preserve_vib:
        melody, vibrato = preserve_vibrato(f0, sr_hop=sr_hop, sr=sr)
    else:
        melody, vibrato = hz_to_cents(f0), np.zeros_like(f0)

    snapped = _snap_cents_to_scale(melody, scale_semitones)
    blended = (1.0 - strength) * melody + strength * snapped

    # Smoothing in frame units.
    frame_sr = sr / sr_hop
    win = max(1, int(round(smooth_ms * 1e-3 * frame_sr)))
    blended = _moving_average(blended, win)

    tuned_cents = blended + vibrato
    return cents_to_hz(tuned_cents, voiced=voiced)
