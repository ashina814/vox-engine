"""1D U-Net diffusion decoder for mel-spectrograms.

Predicts ``v`` (v-prediction) given a noisy mel and a frame-aligned conditioning
tensor. Time step is injected via a sinusoidal embedding broadcast over T.

Tensor conventions:
    mel_t:  (B, n_mels, T)
    t:      (B,) long, in [1, num_steps]
    cond:   (B, hidden, T) — output of ConditionAggregator (+ Whisper branch)
    return: (B, n_mels, T) — v prediction
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard sinusoidal time embedding. t: (B,), out: (B, dim)."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    """Conv1d residual block with time + condition injection."""

    def __init__(self, channels: int, cond_channels: int, time_dim: int, kernel: int = 3) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.time_proj = nn.Linear(time_dim, channels)
        self.cond_proj = nn.Conv1d(cond_channels, channels, kernel_size=1)

    def forward(self, x: Tensor, t_emb: Tensor, cond: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = h + self.cond_proj(cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


@dataclass
class DecoderConfig:
    n_mels: int = 128
    hidden: int = 256
    cond_dim: int = 256
    num_blocks: int = 8
    kernel_size: int = 3
    time_dim: int = 256


class DiffusionDecoder(nn.Module):
    """Conv1d U-Net-style decoder predicting v.

    All convolutions are length-preserving (kernel/2 padding), so T is unchanged
    end-to-end — no downsampling. This keeps the implementation small enough to
    smoke-test on CPU while remaining a faithful Shallow Diffusion backbone.
    """

    def __init__(self, cfg: DecoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c = self.cfg

        self.time_mlp = nn.Sequential(
            nn.Linear(c.time_dim, c.time_dim * 2),
            nn.SiLU(),
            nn.Linear(c.time_dim * 2, c.time_dim),
        )

        self.input_proj = nn.Conv1d(c.n_mels, c.hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    channels=c.hidden,
                    cond_channels=c.cond_dim,
                    time_dim=c.time_dim,
                    kernel=c.kernel_size,
                )
                for _ in range(c.num_blocks)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, c.hidden),
            nn.SiLU(),
            nn.Conv1d(c.hidden, c.n_mels, kernel_size=1),
        )

    def forward(self, mel_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        if cond.shape[-1] != mel_t.shape[-1]:
            cond = F.interpolate(cond, size=mel_t.shape[-1], mode="linear", align_corners=False)

        t_emb = sinusoidal_embedding(t, self.cfg.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(mel_t)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        return self.output_proj(h)
