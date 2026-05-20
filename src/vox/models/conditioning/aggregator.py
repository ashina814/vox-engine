"""Condition aggregator: collapse all conditioning streams into a single tensor.

Outputs ``(B, hidden, T)`` consumed by the diffusion decoder.

Streams:
  content:   (B, 768, T)              ContentVec features, time-aligned to T_mel
  f0:        (B, T)                   Hz, 0 on unvoiced frames
  uv:        (B, T)                   bool/float, 1 = voiced
  loudness:  (B, T)                   A-weighted RMS
  style_id:  (B,)                     discrete style index → embedding
  ref_mel:   (B, n_mels, T_ref) | None  optional reference for GST style vector

The frame-wise streams are projected to ``hidden`` and concatenated additively;
the utterance-level style vector (style embedding + optional GST) is broadcast
across time and used as the FiLM modulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vox.models.conditioning.gst import GlobalStyleTokens


@dataclass
class AggregatorConfig:
    hidden: int = 256
    content_dim: int = 768
    n_styles: int = 3
    n_speakers: int = 1
    gst_num_tokens: int = 10
    gst_num_heads: int = 8
    n_mels: int = 128
    use_gst: bool = True


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: y = gamma * x + beta."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(hidden, 2 * hidden)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        # x: (B, hidden, T)  style: (B, hidden)
        gamma_beta = self.to_gamma_beta(style)  # (B, 2*hidden)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class ConditionAggregator(nn.Module):
    """All-in-one conditioning encoder.

    Forward:
        content:  (B, content_dim, T)
        f0:       (B, T)
        uv:       (B, T) (bool or float)
        loudness: (B, T)
        style_id: (B,) long
        ref_mel:  (B, n_mels, T_ref) or None
        speaker_id: (B,) long or None

    Returns:
        cond:     (B, hidden, T)
    """

    def __init__(self, cfg: AggregatorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AggregatorConfig()
        H = self.cfg.hidden

        self.content_proj = nn.Conv1d(self.cfg.content_dim, H, kernel_size=1)
        self.f0_proj = nn.Conv1d(1, H, kernel_size=1)
        self.loudness_proj = nn.Conv1d(1, H, kernel_size=1)
        self.uv_gate = nn.Conv1d(1, H, kernel_size=1)
        self.speaker_emb = nn.Embedding(self.cfg.n_speakers, H)
        self.style_emb = nn.Embedding(self.cfg.n_styles, H)

        if self.cfg.use_gst:
            self.gst = GlobalStyleTokens(
                num_tokens=self.cfg.gst_num_tokens,
                hidden=H,
                num_heads=self.cfg.gst_num_heads,
                n_mels=self.cfg.n_mels,
            )
        else:
            self.gst = None

        self.film = FiLMBlock(H)
        # Final mixing conv after additive fusion + FiLM.
        self.out_proj = nn.Conv1d(H, H, kernel_size=1)

    @staticmethod
    def _f1d(x: Tensor) -> Tensor:
        """(B, T) → (B, 1, T) float."""
        return x.float().unsqueeze(1)

    def forward(
        self,
        content: Tensor,
        f0: Tensor,
        uv: Tensor,
        loudness: Tensor,
        style_id: Tensor,
        ref_mel: Tensor | None = None,
        speaker_id: Tensor | None = None,
        style_vec: Tensor | None = None,
    ) -> Tensor:
        """If ``style_vec`` is provided, the discrete style_emb lookup is bypassed.

        This lets the inference pipeline blend embeddings (SLERP / barycentric)
        across discrete style ids and pass the result in directly. ``style_id``
        is still required for shape — and for any downstream module that
        branches on it (e.g. WhisperAwareConditioning).
        """
        B, _, T = content.shape

        # Frame-wise streams (additive fusion at T resolution).
        c = self.content_proj(content)
        c = c + self.f0_proj(self._f1d(f0))
        c = c + self.loudness_proj(self._f1d(loudness))
        c = c * torch.sigmoid(self.uv_gate(self._f1d(uv)))  # voiced gate

        # Utterance-level style vector.
        if style_vec is None:
            style_vec = self.style_emb(style_id.long())  # (B, H)
        if speaker_id is None:
            spk = self.speaker_emb(torch.zeros(B, dtype=torch.long, device=c.device))
        else:
            spk = self.speaker_emb(speaker_id.long())
        style_vec = style_vec + spk
        if self.gst is not None and ref_mel is not None:
            style_vec = style_vec + self.gst(ref_mel)

        c = self.film(c, style_vec)
        return self.out_proj(c)
