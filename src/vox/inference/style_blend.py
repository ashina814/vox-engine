"""SLERP-based blending of discrete style embeddings.

Used by the inference pipeline to mix the learned style embedding rows
(``model.aggregator.style_emb.weight``) under user-controlled barycentric
weights, then pass the resulting vector as ``style_vec`` to the model.

Pure spherical interpolation gives more musical transitions than linear mixing
because the FiLM modulator effectively scales by direction; linear mixing
collapses the magnitude near the equidistant point.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def slerp(v0: Tensor, v1: Tensor, alpha: float, eps: float = 1e-6) -> Tensor:
    """Spherical linear interpolation along the last dimension.

    Falls back to a stable linear mix when the two vectors are nearly
    collinear or identical (the SLERP formula divides by sin(theta)).
    """
    if alpha <= 0.0:
        return v0
    if alpha >= 1.0:
        return v1

    n0 = v0.norm(dim=-1, keepdim=True).clamp(min=eps)
    n1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)
    u0 = v0 / n0
    u1 = v1 / n1
    dot = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # When sin_theta is tiny the SLERP formula is numerically unstable;
    # use direct linear interpolation in that regime.
    near_collinear = sin_theta.abs() < eps
    lin_mix = (1.0 - alpha) * v0 + alpha * v1
    slerp_dir = (
        torch.sin((1.0 - alpha) * theta) / sin_theta * u0
        + torch.sin(alpha * theta) / sin_theta * u1
    )
    # Restore the (interpolated) magnitude so SLERP behaves on non-unit inputs.
    mag = (1.0 - alpha) * n0 + alpha * n1
    slerp_out = slerp_dir * mag
    return torch.where(near_collinear.expand_as(lin_mix), lin_mix, slerp_out)


def slerp_barycentric(style_vectors: Sequence[Tensor], weights: Sequence[float]) -> Tensor:
    """Weighted SLERP over an arbitrary number of vectors (3+ via folding).

    Strategy: sort by weight, take the heaviest pair, SLERP them with their
    relative weight ratio, repeat. With three weights this matches the design
    spec's "重み順にペアで SLERP を畳む".

    Args:
        style_vectors: list of (D,) or (B, D) tensors, all the same shape.
        weights: same length as ``style_vectors``; non-negative, sums to > 0.

    Returns:
        Tensor with the same shape as a single style vector.
    """
    if len(style_vectors) != len(weights):
        raise ValueError("style_vectors and weights must have the same length")
    if len(style_vectors) == 0:
        raise ValueError("Need at least one style vector")

    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if total <= 0:
        raise ValueError("Weights must have positive sum")
    w = [x / total for x in w]

    # One-hot shortcut: skip SLERP for numerical exactness.
    if any(abs(x - 1.0) < 1e-9 for x in w):
        return style_vectors[w.index(max(w))]

    # Fold pairwise from heaviest to lightest.
    pairs = sorted(zip(w, style_vectors), key=lambda p: p[0], reverse=True)
    cur_w, cur_v = pairs[0]
    for next_w, next_v in pairs[1:]:
        if next_w == 0.0:
            continue
        # SLERP alpha = next_w / (cur_w + next_w)
        alpha = next_w / (cur_w + next_w)
        cur_v = slerp(cur_v, next_v, alpha)
        cur_w = cur_w + next_w
    return cur_v
