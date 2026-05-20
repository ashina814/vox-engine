"""VOX trainer: drives ``VoxModel.training_step`` with logging + checkpointing.

Intentionally minimal — just enough to run W3 smoke training on Colab T4 and
plug into RunPod runs later. Validation metric implementations (MCD / F0RMSE /
UV-err) live in ``vox.evaluation`` and are dispatched by the BenchmarkRunner
for full eval; here we only track scalar losses on val batches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from vox.training.logging import Logger, StdoutLogger
from vox.training.optim import OptimConfig, build_optimizer, build_scheduler


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    log_interval: int = 100
    val_interval: int = 1000
    ckpt_interval: int = 5000
    grad_clip: float = 1.0
    val_batches: int = 10
    ckpt_dir: Path = Path("checkpoints")
    optim: OptimConfig = field(default_factory=OptimConfig)


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class VoxTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        logger: Logger | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or StdoutLogger()
        self.optimizer = build_optimizer(
            (p for p in self.model.parameters() if p.requires_grad), cfg.optim
        )
        self.scheduler = build_scheduler(self.optimizer, cfg.optim)
        self.global_step = 0
        self.cfg.ckpt_dir = Path(self.cfg.ckpt_dir)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict) -> dict[str, torch.Tensor]:
        batch = _move_to_device(batch, self.device)
        losses = self.model.training_step(batch)

        total = losses["total_loss"]
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return losses

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        for batch in islice(self.val_loader, self.cfg.val_batches):
            batch = _move_to_device(batch, self.device)
            losses = self.model.training_step(batch)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        self.model.train()
        if n == 0:
            return {}
        return {f"val/{k}": v / n for k, v in sums.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, tag: str = "step") -> Path:
        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.ckpt_dir / f"{tag}_{step:08d}.pt"
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = int(ckpt["step"])
        return self.global_step

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.model.train()
        train_iter = iter(self.train_loader)
        while self.global_step < self.cfg.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            losses = self.training_step(batch)
            self.global_step += 1

            if self.global_step % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log(
                    {**{k: float(v) for k, v in losses.items()}, "lr": lr},
                    self.global_step,
                )

            if self.val_loader is not None and self.global_step % self.cfg.val_interval == 0:
                self.logger.log(self.validate(), self.global_step)

            if self.global_step % self.cfg.ckpt_interval == 0:
                self.save_checkpoint(self.global_step)

        self.logger.close()
