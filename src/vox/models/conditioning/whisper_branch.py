"""Whisper-aware conditioning side-branch.

Whisper singing is acoustically distinct (noise-dominated, F0 unstable). A
dedicated residual path is added on top of the base conditioning, gated by:

    1) style_id == WHISPER_ID (per-sample, broadcast over time), and
    2) (1 - uv): the residual fires strongest on unvoiced frames.

This avoids polluting non-whisper styles while still letting the decoder lean
on the residual where it actually matters.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


WHISPER_ID = 1  # configs/* treat 1 as whisper


class WhisperAwareConditioning(nn.Module):
    """Conditional residual branch keyed on style_id == WHISPER_ID.

    Args:
        hidden: matches ConditionAggregator's hidden dim.
        whisper_id: which discrete style id triggers the branch.

    Forward:
        base_cond: (B, hidden, T)
        uv:        (B, T) bool/float, 1 on voiced frames
        style_id:  (B,) long
    Returns:
        (B, hidden, T) — base_cond + (gated whisper residual)
    """

    def __init__(self, hidden: int = 256, whisper_id: int = WHISPER_ID) -> None:
        super().__init__()
        self.whisper_id = whisper_id
        self.residual = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, base_cond: Tensor, uv: Tensor, style_id: Tensor) -> Tensor:
        B, _, T = base_cond.shape
        whisper_mask = (style_id == self.whisper_id).float().view(B, 1, 1)  # (B,1,1)
        if whisper_mask.sum() == 0:
            return base_cond  # no whisper samples → skip the residual entirely

        uv_gate = (1.0 - uv.float()).unsqueeze(1)  # (B, 1, T)
        res = self.residual(base_cond) * uv_gate * whisper_mask
        return base_cond + res
