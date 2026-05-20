"""Optimizer + LR scheduler builders for VOX.

Per the design spec: AdamW (β=(0.9, 0.98), wd=0.01) + cosine schedule with
linear warmup. ``build_optimizer`` and ``build_scheduler`` are thin enough to
be driven straight from a hydra config block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class OptimConfig:
    lr: float = 2.0e-4
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100_000
    min_lr_ratio: float = 0.01  # final LR = lr * min_lr_ratio


def build_optimizer(params, cfg: OptimConfig) -> AdamW:
    """AdamW with the design-spec hyper-parameters."""
    return AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)


def cosine_with_warmup_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.0):
    """Return a LambdaLR multiplier that goes linear→1→cosine→min_lr_ratio."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if step >= max_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return lr_lambda


def build_scheduler(optimizer: Optimizer, cfg: OptimConfig) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=cosine_with_warmup_lambda(cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio),
    )
