"""NSF-HiFiGAN vocoder wrapper.

The real checkpoint is downloaded by ``scripts/download_pretrained.py`` and
loaded here. For Phase A we keep the wrapper thin and the heavy weights
optional — when no checkpoint is available (CI, smoke tests), we fall back to
a tiny placeholder generator with the same I/O signature so the rest of the
pipeline remains exercisable.

Real I/O contract (NSF-HiFiGAN):
    mel: (B, n_mels, T_mel) — normalized log-mel
    f0:  (B, T_mel)         — Hz, 0 on unvoiced frames
    →    (B, T_wav)         — float32 waveform at sr=44100, T_wav = T_mel * hop
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class _PlaceholderGenerator(nn.Module):
    """Tiny mel→wav generator used when the real checkpoint is unavailable.

    Not a real vocoder — only exists so VoxModel can run end-to-end in tests
    and on machines without the NSF-HiFiGAN download.
    """

    def __init__(self, n_mels: int = 128, hop: int = 512) -> None:
        super().__init__()
        self.hop = hop
        # 1-layer transposed conv upsamples T_mel → T_wav.
        self.up = nn.ConvTranspose1d(
            in_channels=n_mels + 1,  # mel + f0
            out_channels=1,
            kernel_size=hop * 2,
            stride=hop,
            padding=hop // 2,
        )

    def forward(self, mel: Tensor, f0: Tensor) -> Tensor:
        x = torch.cat([mel, f0.unsqueeze(1)], dim=1)  # (B, n_mels+1, T_mel)
        wav = self.up(x).squeeze(1)  # (B, T_wav approx)
        return torch.tanh(wav)


class NSFHifiGANWrapper(nn.Module):
    """Loads NSF-HiFiGAN or falls back to a placeholder generator.

    Args:
        ckpt_path: optional path to a NSF-HiFiGAN .ckpt / .pt file.
        n_mels, hop: must match MelExtractor's spec.
        freeze: if True, params have requires_grad=False (default).
    """

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        n_mels: int = 128,
        hop: int = 512,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self.n_mels = n_mels
        self.hop = hop

        if self.ckpt_path is not None and self.ckpt_path.exists():
            self.model = self._load_real(self.ckpt_path)
            self.is_placeholder = False
        else:
            self.model = _PlaceholderGenerator(n_mels=n_mels, hop=hop)
            self.is_placeholder = True

        self.set_frozen(freeze)

    def _load_real(self, path: Path) -> nn.Module:
        # Real NSF-HiFiGAN integration is implemented when the checkpoint is
        # actually downloaded (scripts/download_pretrained.py). The wrapper
        # surface stays identical to the placeholder.
        raise NotImplementedError(
            "Real NSF-HiFiGAN loader is wired in scripts/download_pretrained.py; "
            "Phase A smoke tests use the placeholder generator."
        )

    def set_frozen(self, frozen: bool) -> None:
        for p in self.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.eval()

    def forward(self, mel: Tensor, f0: Tensor) -> Tensor:
        """(B, n_mels, T_mel), (B, T_mel) → (B, T_wav)."""
        if mel.shape[1] != self.n_mels:
            raise ValueError(f"Expected n_mels={self.n_mels}, got {mel.shape[1]}")
        if f0.shape[-1] != mel.shape[-1]:
            raise ValueError(f"f0 T ({f0.shape[-1]}) must equal mel T ({mel.shape[-1]})")
        return self.model(mel, f0)
