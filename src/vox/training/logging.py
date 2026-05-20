"""Pluggable logger abstraction.

The Trainer only uses ``Logger`` — concrete backends (stdout, TensorBoard,
wandb) are lazy-imported so test environments without optional deps still work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Logger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def log_audio(self, name: str, audio: np.ndarray, sr: int, step: int) -> None: ...
    def log_spectrogram(self, name: str, mel: np.ndarray, step: int) -> None: ...
    def close(self) -> None: ...


class StdoutLogger:
    """Bare-bones logger that prints scalars. Audio/spectrograms are dropped."""

    def __init__(self, name: str = "vox") -> None:
        self.name = name

    def log(self, metrics: dict[str, float], step: int) -> None:
        parts = [f"{k}={float(v):.4g}" for k, v in metrics.items()]
        print(f"[{self.name}][step={step}] " + " ".join(parts))

    def log_audio(self, name: str, audio: np.ndarray, sr: int, step: int) -> None:
        print(f"[{self.name}][step={step}] (audio {name!r}: {len(audio)} samples @ {sr} Hz)")

    def log_spectrogram(self, name: str, mel: np.ndarray, step: int) -> None:
        print(f"[{self.name}][step={step}] (spectrogram {name!r}: shape={mel.shape})")

    def close(self) -> None:
        pass


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter  # lazy

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step)

    def log_audio(self, name: str, audio: np.ndarray, sr: int, step: int) -> None:
        self.writer.add_audio(name, audio, step, sample_rate=sr)

    def log_spectrogram(self, name: str, mel: np.ndarray, step: int) -> None:
        # Expect mel in (n_mels, T); add_image needs (C, H, W) or (H, W).
        self.writer.add_image(name, mel[np.newaxis, ...], step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    def __init__(self, project: str = "vox", run_name: str | None = None) -> None:
        import wandb  # lazy

        self._wandb = wandb
        self.run = wandb.init(project=project, name=run_name, reinit=True)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    def log_audio(self, name: str, audio: np.ndarray, sr: int, step: int) -> None:
        self._wandb.log({name: self._wandb.Audio(audio, sample_rate=sr)}, step=step)

    def log_spectrogram(self, name: str, mel: np.ndarray, step: int) -> None:
        self._wandb.log({name: self._wandb.Image(mel)}, step=step)

    def close(self) -> None:
        self.run.finish()


def build_logger(kind: str = "stdout", **kwargs) -> Logger:
    """Factory dispatch on a config string."""
    if kind == "stdout":
        return StdoutLogger(**kwargs)
    if kind == "tensorboard":
        return TensorBoardLogger(**kwargs)
    if kind == "wandb":
        return WandbLogger(**kwargs)
    raise ValueError(f"Unknown logger kind: {kind!r}")
