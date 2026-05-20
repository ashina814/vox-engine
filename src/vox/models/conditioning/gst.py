"""Global Style Tokens (Wang et al., 2018) for reference-driven style embedding.

Reference encoder maps (B, n_mels, T_ref) → (B, hidden) via a stack of
Conv2D + GRU, then multi-head attention is taken over a bank of learnable
style tokens to produce a single style vector per utterance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReferenceEncoder(nn.Module):
    """6-layer Conv2D + GRU encoder, mel → utterance embedding.

    Input:  (B, n_mels, T_ref)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        n_mels: int = 128,
        out_dim: int = 256,
        conv_channels: tuple[int, ...] = (32, 32, 64, 64, 128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_c = 1
        for c in conv_channels:
            layers += [
                nn.Conv2d(in_c, c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            in_c = c
        self.conv = nn.Sequential(*layers)
        # After 6 stride-2 convs over the mel axis: ceil(n_mels / 2^6)
        mel_after = n_mels
        for _ in conv_channels:
            mel_after = (mel_after + 1) // 2
        self.gru = nn.GRU(input_size=in_c * mel_after, hidden_size=out_dim, batch_first=True)

    def forward(self, mel: Tensor) -> Tensor:
        # (B, n_mels, T_ref) → (B, 1, n_mels, T_ref)
        x = mel.unsqueeze(1)
        x = self.conv(x)  # (B, C, n_mels', T')
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * M)  # (B, T', C*M)
        _, h = self.gru(x)  # h: (1, B, out_dim)
        return h.squeeze(0)


class GlobalStyleTokens(nn.Module):
    """Attention over a learnable token bank, keyed by a reference embedding.

    Args:
        num_tokens: size of the learnable token bank.
        hidden: token / output dimensionality.
        num_heads: attention heads.
        n_mels: mel bins of the reference input.

    Forward:
        ref_mel: (B, n_mels, T_ref) → style: (B, hidden)
    """

    def __init__(
        self,
        num_tokens: int = 10,
        hidden: int = 256,
        num_heads: int = 8,
        n_mels: int = 128,
    ) -> None:
        super().__init__()
        self.ref_encoder = ReferenceEncoder(n_mels=n_mels, out_dim=hidden)
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden) * 0.5)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.hidden = hidden

    def forward(self, ref_mel: Tensor) -> Tensor:
        # Query = reference embedding (B, 1, hidden); keys/values = token bank (B, N, hidden).
        ref_emb = self.ref_encoder(ref_mel).unsqueeze(1)  # (B, 1, hidden)
        B = ref_emb.shape[0]
        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1)  # (B, N, hidden)
        style, _ = self.attn(ref_emb, tokens, tokens)
        return style.squeeze(1)  # (B, hidden)
