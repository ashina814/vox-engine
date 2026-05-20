"""Training losses for the VOX diffusion model.

Each loss accepts an optional ``mask`` of shape ``(B, T)`` so padded frames in
collated batches don't pollute gradients. The masked mean denominator is
``mask.sum() * feature_dim`` so loss magnitudes stay comparable to the unmasked
``.mean()`` reduction.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(diff_sq: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over (B, C, T) with optional time-mask (B, T)."""
    if mask is None:
        return diff_sq.mean()
    mask_f = mask.float().unsqueeze(1)  # (B, 1, T)
    denom = mask_f.sum().clamp(min=1.0) * diff_sq.shape[1]
    return (diff_sq * mask_f).sum() / denom


def diffusion_loss(v_pred: Tensor, v_target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Masked L2 between predicted and target v."""
    return _masked_mean((v_pred - v_target).pow(2), mask)


def mel_l1_loss(mel_pred: Tensor, mel_target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Auxiliary L1 between reconstructed mel and ground-truth mel."""
    diff = (mel_pred - mel_target).abs()
    if mask is None:
        return diff.mean()
    mask_f = mask.float().unsqueeze(1)
    denom = mask_f.sum().clamp(min=1.0) * diff.shape[1]
    return (diff * mask_f).sum() / denom


def f0_consistency_loss(
    mel_pred: Tensor,
    f0_target: Tensor,
    f0_extractor,
    mask: Tensor | None = None,
) -> Tensor:
    """Re-extract F0 from the predicted mel-derived waveform and compare.

    ``f0_extractor`` is expected to be a callable ``(B, n_mels, T) -> (B, T)``
    that maps mel directly to an F0 estimate (avoids the costly vocoder step
    during training). When not provided, we fall back to a no-op zero loss.

    The default Phase A training run will pass ``None`` here and rely on the
    diffusion loss alone — this hook exists so the loss can be swapped in later
    without churning the trainer API.
    """
    if f0_extractor is None:
        return torch.zeros((), device=mel_pred.device)
    f0_pred = f0_extractor(mel_pred)
    diff_sq = (f0_pred - f0_target).pow(2)
    if mask is None:
        return diff_sq.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (diff_sq * mask_f).sum() / denom
