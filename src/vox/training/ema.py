"""Exponential Moving Average of model weights.

Standard technique across modern generative training (Stable Diffusion,
DiT, Flow Matching repos) — maintaining an EMA copy of the parameters and
using *those* at inference time consistently buys 5-10% quality with no
extra forward passes during training.

Update rule per training step:
    ema_param <- decay * ema_param + (1 - decay) * current_param

Decay choice as a function of dataset / step budget:
    decay = 0.999  → window ~  1,000 steps. Good default for >50k step runs.
    decay = 0.99   → window ~    100 steps. Better for small-dataset runs
                     (e.g. Phase B's 5.5h voice → ~5k-10k steps).
    decay = 0.9999 → window ~ 10,000 steps. Stable Diffusion 2 setting.

Warmup: linearly ramp from 0 → target decay over ``warmup_steps``. Avoids
the artefact where an unwarmed EMA in the first hundred steps tracks
nearly identical to the live weights (low decay) — useful when you start
EMA from a randomly-initialised model.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintains a shadow copy of model parameters under EMA.

    Args:
        model: source of trainable parameters; only ``requires_grad=True``
            tensors are tracked.
        decay: target decay (max effective decay after warmup).
        warmup_steps: linearly ramp ``effective_decay`` from 0 to ``decay``
            over this many calls to ``update``. Set to 0 (default) for
            constant decay from the first step.

    Usage:
        ema = ExponentialMovingAverage(model, decay=0.999, warmup_steps=1000)
        # ... after each optimizer.step():
        ema.update(model)
        # ... at evaluation / inference:
        with ema.swap_in(model):
            outputs = model(...)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        # Snapshot trainable params + persistent buffers.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()
        self._original: dict[str, torch.Tensor] = {}

    def current_decay(self) -> float:
        """The decay value that *will be applied* on the next ``update``."""
        if self.warmup_steps <= 0:
            return self.decay
        ramp = min(1.0, (self._step + 1) / float(self.warmup_steps + 1))
        return self.decay * ramp

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.current_decay()
        self._step += 1
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow[name]
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict:
        """Return shadow tensors + step counter for full restore.

        Layout: ``{ "_step": int, "_shadow": {name: tensor, ...} }``.
        Backward-compatible with the old flat ``{name: tensor}`` layout —
        ``load_state_dict`` handles both.
        """
        return {
            "_step": self._step,
            "_shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        # Accept both the new nested layout and the legacy flat layout that
        # earlier checkpoints used.
        if "_shadow" in state:
            shadow = state["_shadow"]
            self._step = int(state.get("_step", 0))
        else:
            shadow = state
        for k in self.shadow:
            if k in shadow:
                self.shadow[k] = shadow[k].to(self.shadow[k].device).clone()

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
