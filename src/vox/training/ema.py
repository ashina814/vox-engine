"""Exponential Moving Average of model weights.

Standard technique across modern generative training (Stable Diffusion,
DiT, Flow Matching repos) — maintaining an EMA copy of the parameters and
using *those* at inference time consistently buys 5-10% quality with no
extra forward passes during training. We're paying half a day of code for
free quality.

Update rule per training step:
    ema_param <- decay * ema_param + (1 - decay) * current_param

Default decay 0.999 → effective averaging window ~ 1000 steps.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintains a shadow copy of model parameters under EMA.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999)
        # ... after each optimizer.step():
        ema.update(model)
        # ... at evaluation / inference:
        with ema.swap_in(model):
            outputs = model(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        # Snapshot trainable params + persistent buffers.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
        self._original: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # Allow loading EMA from a possibly-larger checkpoint state_dict.
        for k in self.shadow:
            if k in state:
                self.shadow[k] = state[k].to(self.shadow[k].device).clone()

    def to(self, device) -> "ExponentialMovingAverage":
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    # ------------------------------------------------------------------
    # Inference helpers — temporarily swap EMA weights into the model
    # ------------------------------------------------------------------

    class _SwapContext:
        def __init__(self, ema: "ExponentialMovingAverage", model: nn.Module) -> None:
            self.ema = ema
            self.model = model

        def __enter__(self):
            self.ema._original = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if p.requires_grad and name in self.ema.shadow
            }
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in self.ema.shadow:
                        p.copy_(self.ema.shadow[name])
            return self.model

        def __exit__(self, *exc):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in self.ema._original:
                        p.copy_(self.ema._original[name])
            self.ema._original = {}

    def swap_in(self, model: nn.Module) -> "_SwapContext":
        """Context manager: model uses EMA weights inside the ``with`` block."""
        return self._SwapContext(self, model)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Permanently copy EMA weights into ``model`` (e.g. before saving)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])

    def clone_model(self, model: nn.Module) -> nn.Module:
        """Return a deep copy of ``model`` with EMA weights loaded in."""
        ema_model = copy.deepcopy(model)
        self.copy_to(ema_model)
        return ema_model
