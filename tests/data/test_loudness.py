import numpy as np
import torch

from vox.data.features.loudness import a_weighted_rms
from vox.data.features.mel import MelExtractor


def test_output_shape(sine_wav):
    loudness = a_weighted_rms(sine_wav)
    mel_frames = MelExtractor().frames(len(sine_wav))
    # Loudness frame count matches mel frame count
    assert loudness.shape == (
        mel_frames,
    ), f"loudness shape {loudness.shape}, expected ({mel_frames},)"


def test_non_negative(sine_wav):
    loudness = a_weighted_rms(sine_wav)
    assert (loudness >= 0).all()


def test_silence_near_zero():
    silence = torch.zeros(44_100 * 3)
    loudness = a_weighted_rms(silence)
    assert loudness.max().item() < 1e-6


def test_louder_signal_higher_rms():
    t = np.linspace(0, 3.0, 44_100 * 3, dtype=np.float32)
    quiet = torch.from_numpy(0.1 * np.sin(2 * np.pi * 440 * t))
    loud = torch.from_numpy(0.9 * np.sin(2 * np.pi * 440 * t))
    assert a_weighted_rms(loud).mean() > a_weighted_rms(quiet).mean()
