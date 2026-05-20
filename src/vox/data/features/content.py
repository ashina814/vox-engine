"""ContentVec-768 feature extractor (frozen, not trained)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


class ContentVecExtractor:
    """Extract ContentVec-768 features from audio.

    Uses HuggingFace transformers to load the lengyue233/content-vec-best
    checkpoint. The model is always frozen (eval mode, no grad).

    Args:
        ckpt: HuggingFace model ID or local path.
              Default: 'lengyue233/content-vec-best'
        layer: transformer layer index to extract (1-indexed). Default: 12.
        device: target device.
    """

    DEFAULT_CKPT = "lengyue233/content-vec-best"
    # ContentVec runs at 16 kHz internally; torchaudio resamples for us.
    _MODEL_SR = 16_000

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        layer: int = 12,
        device: str | torch.device = "cpu",
    ) -> None:
        self.layer = layer
        self.device = torch.device(device)
        self._model = self._load(str(ckpt))

    def _load(self, ckpt: str):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        model = HubertModel.from_pretrained(ckpt, output_hidden_states=True)
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.no_grad()
    def __call__(self, wav: Tensor, src_sr: int = 44_100) -> Tensor:
        """Extract content features.

        Args:
            wav: (T_wav,) float32 audio at src_sr.
            src_sr: sample rate of the input wav.

        Returns:
            content: (768, T_content) float32.
                     T_content ≈ T_wav / (src_sr / MODEL_SR * 320).
                     Caller must interpolate to T_mel if needed.
        """
        import torchaudio

        # Resample to 16 kHz if needed
        if src_sr != self._MODEL_SR:
            resampler = torchaudio.transforms.Resample(src_sr, self._MODEL_SR)
            wav = resampler(wav)

        inputs = self._processor(
            wav.numpy(),
            sampling_rate=self._MODEL_SR,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        out = self._model(input_values, output_hidden_states=True)
        # hidden_states: tuple of (1, T_content, 768) for each layer
        hidden = out.hidden_states[self.layer]  # (1, T_content, 768)
        return hidden.squeeze(0).T.cpu()  # (768, T_content)

    @staticmethod
    def interpolate_to_mel(content: Tensor, t_mel: int) -> Tensor:
        """Linearly interpolate content to match mel frame count.

        Args:
            content: (768, T_content)
            t_mel: target number of mel frames

        Returns:
            (768, t_mel)
        """
        return F.interpolate(
            content.unsqueeze(0),  # (1, 768, T_content)
            size=t_mel,
            mode="linear",
            align_corners=False,
        ).squeeze(0)  # (768, t_mel)
