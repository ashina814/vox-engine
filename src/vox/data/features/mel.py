from __future__ import annotations

import torch
import torchaudio
import torchaudio.transforms as T
from torch import Tensor


class MelExtractor:
    """Mel-spectrogram extractor with fixed VOX audio spec.

    Converts (T_wav,) float32 tensor → (n_mels, T_mel) float32 in [0, 1].
    """

    SR = 44_100
    N_MELS = 128
    N_FFT = 2048
    HOP = 512
    WIN = 2048
    DB_MIN = -80.0
    DB_MAX = 0.0

    def __init__(
        self,
        sr: int = SR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop: int = HOP,
        win: int = WIN,
    ) -> None:
        self.sr = sr
        self.hop = hop
        self._mel = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0,
        )
        self._to_db = T.AmplitudeToDB(stype="power", top_db=-self.DB_MIN)

    def __call__(self, wav: Tensor) -> Tensor:
        """(T_wav,) → (n_mels, T_mel), values in [0, 1]."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, T_wav)
        mel = self._mel(wav).squeeze(0)  # (n_mels, T_mel)
        mel_db = self._to_db(mel)  # dB, range approx [DB_MIN, DB_MAX]
        mel_norm = (mel_db - self.DB_MIN) / (self.DB_MAX - self.DB_MIN)
        return mel_norm.clamp(0.0, 1.0)

    def frames(self, n_samples: int) -> int:
        """Number of mel frames for a given number of audio samples.

        Matches torchaudio MelSpectrogram with center=True (default):
        T_mel = n_samples // hop + 1.
        """
        return n_samples // self.hop + 1
