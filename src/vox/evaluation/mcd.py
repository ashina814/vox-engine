"""Mel Cepstral Distortion with DTW time alignment.

Phase A uses MFCC as a stand-in for true MGC (mel-generalized cepstrum) so
this module has no pysptk dependency. The scale is therefore not numerically
identical to the canonical MCD literature, but the *relative* values track
quality the same way and the metric still satisfies the design contract
(identical waveforms → ~0, dissimilar → larger value).
"""
from __future__ import annotations

import numpy as np

# Pre-factor that converts squared-cepstral-distance to the canonical MCD unit (dB).
_MCD_K = (10.0 / np.log(10.0)) * np.sqrt(2.0)


def _mfcc(wav: np.ndarray, sr: int, n_mcep: int, hop: int = 512, n_fft: int = 2048) -> np.ndarray:
    import librosa

    # n_mfcc returned coefficients include C0; drop it to mimic the standard MCD form.
    mfcc = librosa.feature.mfcc(
        y=wav.astype(np.float32),
        sr=sr,
        n_mfcc=n_mcep + 1,
        n_fft=n_fft,
        hop_length=hop,
    )
    return mfcc[1:]  # (n_mcep, T)


def mel_cepstral_distortion(
    wav_ref: np.ndarray,
    wav_pred: np.ndarray,
    sr: int = 44_100,
    n_mcep: int = 13,
    hop: int = 512,
) -> float:
    """Time-aligned MCD (dB) between two waveforms.

    Frames are aligned via librosa.sequence.dtw on the mfcc cost matrix; the
    final score is the mean per-frame L2 of cepstrum differences along the
    warp path, scaled by the standard MCD factor.
    """
    import librosa

    ref = _mfcc(wav_ref, sr=sr, n_mcep=n_mcep, hop=hop)
    pred = _mfcc(wav_pred, sr=sr, n_mcep=n_mcep, hop=hop)

    # librosa.sequence.dtw expects (n_feats, T).
    _D, wp = librosa.sequence.dtw(X=ref, Y=pred, metric="euclidean")
    pairs = wp[::-1]  # path in increasing time order
    if len(pairs) == 0:
        return 0.0

    diffs = ref[:, pairs[:, 0]] - pred[:, pairs[:, 1]]  # (n_mcep, L)
    per_frame = np.sqrt((diffs ** 2).sum(axis=0))
    return float(_MCD_K * per_frame.mean())
