"""DDIM sampler driving a v-prediction diffusion decoder.

K=50 inference steps as called out in the design spec.
"""
from __future__ import annotations

import torch
from torch import Tensor

from vox.models.diffusion.noise_schedule import NoiseSchedule


class DDIMSampler:
    """Deterministic DDIM sampler with v-prediction.

    Translates v → (x0_pred, eps_pred), then takes the standard DDIM step:
        x_{t_prev} = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps_pred
    """

    def __init__(self, schedule: NoiseSchedule) -> None:
        self.schedule = schedule

    def _v_to_x0_eps(self, x_t: Tensor, v: Tensor, ab: Tensor) -> tuple[Tensor, Tensor]:
        s = ab.sqrt().view(-1, 1, 1)
        s1 = (1.0 - ab).sqrt().view(-1, 1, 1)
        x0 = s * x_t - s1 * v
        eps = s1 * x_t + s * v
        return x0, eps

    @torch.no_grad()
    def sample(
        self,
        decoder,
        cond: Tensor,
        shape: tuple[int, int, int],
        num_steps: int = 50,
        device: torch.device | str | None = None,
    ) -> Tensor:
        device = torch.device(device) if device is not None else cond.device
        B = shape[0]
        x = torch.randn(*shape, device=device)

        # Uniform stride through [1, T].
        ts = torch.linspace(self.schedule.num_steps, 1, num_steps + 1, device=device).long()
        alpha_bar = self.schedule.alpha_bar.to(device)

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            t_next = ts[i + 1]
            ab_cur = alpha_bar[t_cur.long()].clamp(min=1e-8)
            ab_next = alpha_bar[t_next.long()].clamp(min=1e-8).expand(B)

            v = decoder(x, t_cur, cond)
            x0, eps = self._v_to_x0_eps(x, v, ab_cur)
            s_ab_next = ab_next.sqrt().view(-1, 1, 1)
            s_1mab_next = (1.0 - ab_next).sqrt().view(-1, 1, 1)
            x = s_ab_next * x0 + s_1mab_next * eps

        return x
