"""A-weighted loudness (RMS per mel frame)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor


@lru_cache(maxsize=4)
def _a_weighting_coeffs(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """IEC 61672 A-weighting filter via bilinear transform.

    Stable digital IIR for any sr; gain normalized to 0 dB at 1 kHz.
    """
    from scipy.signal import bilinear

    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997

    nums = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20.0)), 0.0, 0.0, 0.0, 0.0]
    dens = np.polymul(
        [1.0, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1.0, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
    )
    dens = np.polymul(np.polymul(dens, [1.0, 2 * np.pi * f3]), [1.0, 2 * np.pi * f2])
    b, a = bilinear(nums, dens, sr)
    return b.astype(np.float64), a.astype(np.float64)


def a_weighted_rms(wav: Tensor, hop: int = 512, win: int = 2048, sr: int = 44_100) -> Tensor:
    """A-weighted RMS energy per mel frame, centered to match MelExtractor.

    Args:
        wav: (T_wav,) float32 audio tensor.
        hop: hop length in samples (must match MelExtractor.hop).
        win: window length in samples.
        sr: sample rate (for A-weight filter design).

    Returns:
        loudness: (T_mel,) float32 tensor, T_mel = T_wav // hop + 1, non-negative.
    """
    from scipy.signal import lfilter

    wav_np = wav.detach().cpu().numpy().astype(np.float64)
    b, a = _a_weighting_coeffs(sr)
    weighted = lfilter(b, a, wav_np)

    # Center-pad to match torchaudio MelSpectrogram(center=True): T_mel = T_wav // hop + 1.
    pad = win // 2
    padded = np.pad(weighted, (pad, pad), mode="reflect")
    n_frames = wav_np.shape[-1] // hop + 1

    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = padded[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(frame * frame)))

    return torch.from_numpy(rms)
