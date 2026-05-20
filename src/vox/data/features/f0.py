"""F0 (fundamental frequency) extractor.

Backend abstraction: torchcrepe is used for W1 validation.
RMVPE can be swapped in by passing backend='rmvpe' once integrated.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


class F0Extractor:
    """Extract F0 and confidence from audio.

    Returns (f0, confidence) in Hz, aligned to mel hop grid.

    Args:
        backend: 'torchcrepe' (default, W1) or 'rmvpe' (Phase B).
        hop: hop length in samples, must match MelExtractor.hop.
        sr: sample rate.
        fmin: minimum F0 in Hz.
        fmax: maximum F0 in Hz.
        threshold: confidence threshold below which frames are unvoiced.
    """

    def __init__(
        self,
        backend: Literal["torchcrepe", "rmvpe"] = "torchcrepe",
        hop: int = 512,
        sr: int = 44_100,
        fmin: float = 50.0,
        fmax: float = 800.0,
        threshold: float = 0.03,
        device: str | torch.device = "cpu",
    ) -> None:
        self.backend = backend
        self.hop = hop
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.device = torch.device(device)

        if backend == "torchcrepe":
            import torchcrepe  # noqa: F401 – validate importable

            self._extract = self._extract_crepe
        elif backend == "rmvpe":
            self._extract = self._extract_rmvpe
        else:
            raise ValueError(f"Unknown F0 backend: {backend!r}")

    def __call__(self, wav: Tensor) -> tuple[Tensor, Tensor]:
        """(T_wav,) → (f0: (T_mel,), confidence: (T_mel,))

        f0 values are in Hz; unvoiced frames have f0 = 0.0.
        """
        if wav.dim() != 1:
            raise ValueError(f"Expected 1-D wav tensor, got shape {wav.shape}")
        return self._extract(wav)

    # ------------------------------------------------------------------
    # torchcrepe backend
    # ------------------------------------------------------------------

    def _extract_crepe(self, wav: Tensor) -> tuple[Tensor, Tensor]:
        import torch.nn.functional as F
        import torchcrepe

        audio = wav.unsqueeze(0).to(self.device)  # (1, T)
        f0, confidence = torchcrepe.predict(
            audio,
            self.sr,
            hop_length=self.hop,
            fmin=self.fmin,
            fmax=self.fmax,
            model="full",
            return_periodicity=True,
            device=self.device,
            batch_size=512,
            decoder=torchcrepe.decode.viterbi,
        )
        f0 = f0.squeeze(0)
        confidence = confidence.squeeze(0)

        # Resample to mel-aligned frame grid (T_wav // hop + 1).
        target = wav.shape[-1] // self.hop + 1
        if f0.shape[-1] != target:
            f0 = F.interpolate(
                f0.view(1, 1, -1), size=target, mode="linear", align_corners=False
            ).view(-1)
            confidence = F.interpolate(
                confidence.view(1, 1, -1), size=target, mode="linear", align_corners=False
            ).view(-1)

        # Zero out unvoiced frames
        f0 = torch.where(confidence >= self.threshold, f0, torch.zeros_like(f0))
        return f0.cpu(), confidence.cpu()

    # ------------------------------------------------------------------
    # RMVPE backend (stub – fill in when integrating RMVPE)
    # ------------------------------------------------------------------

    def _extract_rmvpe(self, wav: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("RMVPE backend not yet integrated")
