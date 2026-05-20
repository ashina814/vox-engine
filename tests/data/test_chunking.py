import numpy as np
import torch
import pytest

from vox.data.chunking import chunk_wav


@pytest.fixture
def sr():
    return 44_100


def _tone(sr: int, duration_s: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(sr: int, duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def test_chunk_pure_silence_returns_empty(sr):
    wav = torch.from_numpy(_silence(sr, 10.0))
    assert chunk_wav(wav, sr) == []


def test_chunk_continuous_tone_splits_at_max(sr):
    """25s tone with no silence → must split into <=10s chunks."""
    wav = torch.from_numpy(_tone(sr, 25.0))
    chunks = chunk_wav(wav, sr, min_s=5.0, max_s=10.0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.duration_s <= 10.0 + 1e-6
        assert c.duration_s >= 5.0


def test_chunk_drops_short_segments(sr):
    """Short tone followed by long silence → too-short chunk dropped."""
    wav = np.concatenate([_tone(sr, 1.0), _silence(sr, 5.0)])
    chunks = chunk_wav(torch.from_numpy(wav), sr, min_s=5.0, max_s=10.0)
    assert chunks == []


def test_chunk_preserves_sample_provenance(sr):
    wav = torch.from_numpy(_tone(sr, 15.0))
    chunks = chunk_wav(wav, sr, min_s=5.0, max_s=10.0)
    for c in chunks:
        assert c.end_sample - c.start_sample == c.num_samples
        assert c.wav.shape == (c.num_samples,)
        assert c.sr == sr
