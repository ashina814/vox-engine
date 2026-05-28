"""Flow Matching schedule + sampler for the mel decoder.

A drop-in replacement for the DDIM diffusion pipeline based on the linear
conditional flow of Lipman et al. (2022) and Liu et al. (2022, "Rectified
Flow"). The key idea: instead of a curved diffusion trajectory, we train the
decoder to predict a *constant-velocity field* along a straight line between
data and noise. Inference then integrates that ODE — typically with just 4
Euler steps — to match what diffusion needs 50+ steps to achieve.

Forward (training):
    x_t = (1 - t) * x_0 + t * noise           with  t ~ Uniform(0, 1]
    v_target = noise - x_0                     (constant in t)

Reverse (sampling):
    Start at x_1 = noise. Repeatedly compute v_pred = decoder(x, t, cond)
    and step  x <- x - dt * v_pred  while marching t: 1 -> 0.

API is intentionally kept close to ``NoiseSchedule`` / ``DDIMSampler`` so
``VoxModel`` can swap between them with a single config flag.
"""

from __future__ import annotations

import torch
from torch import Tensor


class FlowMatchingSchedule:
    """Linear conditional flow schedule (Lipman et al. 2022).

    Args:
        num_steps: training-time t is drawn from a discrete grid of this size
            for compatibility with the existing ``decoder(mel_t, t, cond)``
            signature, which expects integer time embeddings. Sampling can use
            any number of steps independently.
    """

    def __init__(self, num_steps: int = 1000) -> None:
        self.num_steps = num_steps

    def to(self, device) -> "FlowMatchingSchedule":  # noqa: D401 — symmetry with NoiseSchedule
        return self

    def _t_continuous(self, t: Tensor) -> Tensor:
        """Map discrete t in [1, num_steps] to continuous t' in (0, 1]."""
        return t.float() / float(self.num_steps)

    def add_noise(
        self, x0: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (x_t, noise, v_target).

        ``v_target`` is the velocity the decoder should learn to predict.
        Matches the ``NoiseSchedule.add_noise`` contract so the trainer needs
        no changes.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        t_c = self._t_continuous(t).view(-1, 1, 1)
        x_t = (1.0 - t_c) * x0 + t_c * noise
        v_target = noise - x0
        return x_t, noise, v_target

    def sample_random_t(self, batch_size: int, device) -> Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,), device=device)


class FlowMatchingSampler:
    """Euler integrator over the learned velocity field.

    K=4 default steps matches FlashAudio's reported best-quality budget for
    rectified-flow audio generation. The numeric speedup over DDIM K=50 is
    roughly 10x at comparable quality once the model has been trained with
    the matching schedule.
    """

    def __init__(self, schedule: FlowMatchingSchedule) -> None:
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        decoder,
        cond: Tensor,
        shape: tuple[int, int, int],
        num_steps: int = 4,
        device: torch.device | str | None = None,
    ) -> Tensor:
        device = torch.device(device) if device is not None else cond.device
        B = shape[0]
        x = torch.randn(*shape, device=device)

        # Discretise t in [1, num_steps_in_schedule] for the decoder's integer
        # time embedding, but march along K equally spaced points.
        ts = torch.linspace(self.schedule.num_steps, 1, num_steps + 1, device=device).long()
        # Continuous coordinates used to size the Euler step dt.
        ts_c = ts.float() / float(self.schedule.num_steps)

        for i in range(num_steps):
            t_cur = ts[i].expand(B)
            dt = (ts_c[i] - ts_c[i + 1]).item()  # positive: marching t -> 0
            v_pred = decoder(x, t_cur, cond)
            x = x - dt * v_pred  # Euler step along the flow

        return x
