"""Standalone synthetic audio generators (usable outside pytest)."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def sine_wav(freq: float = 440.0, duration_s: float = 5.0, sr: int = 44_100) -> Tensor:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    return torch.from_numpy(np.sin(2 * np.pi * freq * t))


def harmonic_wav(
    f0: float = 220.0, n_harmonics: int = 5, duration_s: float = 5.0, sr: int = 44_100
) -> Tensor:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    wav = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, n_harmonics + 1))
    wav = wav.astype(np.float32)
    wav /= np.abs(wav).max() + 1e-9
    return torch.from_numpy(wav)


def chirp_wav(
    f0_start: float = 100.0, f0_end: float = 400.0, duration_s: float = 5.0, sr: int = 44_100
) -> Tensor:
    from scipy.signal import chirp

    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    wav = chirp(t, f0=f0_start, f1=f0_end, t1=duration_s, method="linear").astype(np.float32)
    return torch.from_numpy(wav)


def make_fake_batch(
    B: int = 2, T_mel: int = 200, n_mels: int = 128
) -> dict[str, torch.Tensor]:
    """Random batch dict passable directly to training_step."""
    return {
        "mel": torch.rand(B, n_mels, T_mel),
        "f0": torch.rand(B, T_mel) * 400 + 100,
        "uv": torch.rand(B, T_mel) > 0.3,
        "content": torch.randn(B, 768, T_mel),
        "loudness": torch.rand(B, T_mel) * 0.1,
        "style_id": torch.zeros(B, dtype=torch.long),
        "mask": torch.ones(B, T_mel, dtype=torch.bool),
    }
