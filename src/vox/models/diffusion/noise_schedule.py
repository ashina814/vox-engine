"""Cosine noise schedule + v-prediction utilities.

Following Karras et al. parameterisation; v-prediction (Salimans & Ho, 2022)
is more numerically stable than epsilon-prediction near both t=0 and t=T.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cosine_alpha_bar(num_steps: int, s: float = 0.008) -> Tensor:
    """alpha_bar(t) = cos((t/T + s) / (1 + s) * pi/2) ** 2 ; t in [0, T]."""
    t = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    alpha_bar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] == 1
    return alpha_bar.float()


class NoiseSchedule:
    """Cosine-scheduled diffusion forward process with v-prediction targets.

    Conventions:
        alpha_bar:  (num_steps + 1,)  cumulative product, alpha_bar[0] == 1
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        v   = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x0
    """

    def __init__(self, num_steps: int = 1000) -> None:
        self.num_steps = num_steps
        self.alpha_bar = cosine_alpha_bar(num_steps)

    def to(self, device) -> "NoiseSchedule":
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def _gather(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (sqrt_ab, sqrt_1_minus_ab) shaped for broadcasting on (B, C, T)."""
        ab = self.alpha_bar.to(t.device)[t.long().clamp(min=0, max=self.num_steps)]
        ab = ab.clamp(min=1e-8, max=1.0)
        s = ab.sqrt().view(-1, 1, 1)
        return s, (1.0 - ab).sqrt().view(-1, 1, 1)

    def add_noise(
        self, x0: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (x_t, noise, v_target)."""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab, s_1mab = self._gather(t)
        x_t = s_ab * x0 + s_1mab * noise
        v = s_ab * noise - s_1mab * x0
        return x_t, noise, v

    def sample_random_t(self, batch_size: int, device) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,), device=device)
