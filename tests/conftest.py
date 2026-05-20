import numpy as np
import pytest
import torch


@pytest.fixture
def sr():
    return 44_100


@pytest.fixture
def hop():
    return 512


@pytest.fixture
def sine_wav(sr):
    """5-second 440 Hz sine wave."""
    t = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False, dtype=np.float32)
    return torch.from_numpy(np.sin(2 * np.pi * 440 * t))


@pytest.fixture
def harmonic_wav(sr):
    """5-second harmonic tone (f0=220, 5 harmonics)."""
    t = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False, dtype=np.float32)
    wav = sum(
        np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6)
    ).astype(np.float32)
    wav /= np.abs(wav).max() + 1e-9
    return torch.from_numpy(wav)


@pytest.fixture
def chirp_wav(sr):
    """5-second linear chirp 100→400 Hz."""
    from scipy.signal import chirp

    t = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False, dtype=np.float32)
    wav = chirp(t, f0=100, f1=400, t1=5.0, method="linear").astype(np.float32)
    return torch.from_numpy(wav)
