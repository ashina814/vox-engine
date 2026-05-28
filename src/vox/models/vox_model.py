"""End-to-end VOX model: conditioning → diffusion mel → vocoder wav.

The ContentVec encoder is loaded externally (it's slow + needs a download), so
this module takes pre-extracted ``content`` features as input. Same goes for
the vocoder: the wrapper handles its own checkpoint and falls back to a
placeholder generator when absent.

Tensor contract on every batch:
    mel:        (B, n_mels, T)
    f0:         (B, T)
    uv:         (B, T)
    content:    (B, 768, T)
    loudness:   (B, T)
    style_id:   (B,) long
    ref_mel:    (B, n_mels, T_ref) | None  (optional, drives GST)
    speaker_id: (B,) long | None
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from vox.models.conditioning.aggregator import AggregatorConfig, ConditionAggregator
from vox.models.conditioning.whisper_branch import WhisperAwareConditioning
from vox.models.diffusion.decoder import DecoderConfig, DiffusionDecoder
from vox.models.diffusion.flow_matching import FlowMatchingSampler, FlowMatchingSchedule
from vox.models.diffusion.noise_schedule import NoiseSchedule
from vox.models.diffusion.sampler import DDIMSampler
from vox.models.vocoder.nsf_hifigan import NSFHifiGANWrapper

ScheduleType = Literal["diffusion", "flow_matching"]


@dataclass
class VoxModelConfig:
    n_mels: int = 128
    hop: int = 512
    hidden: int = 256
    n_styles: int = 3
    whisper_id: int = 1
    diffusion_steps: int = 1000
    schedule_type: ScheduleType = "flow_matching"  # 2025 default: 1-step capable
    aggregator: AggregatorConfig = field(default_factory=lambda: AggregatorConfig())
    decoder: DecoderConfig = field(default_factory=lambda: DecoderConfig())
    vocoder_ckpt: str | None = None


class VoxModel(nn.Module):
    def __init__(self, cfg: VoxModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VoxModelConfig()
        c = self.cfg

        # Make sure the agg/decoder configs agree with the top-level dims.
        c.aggregator.hidden = c.hidden
        c.aggregator.n_styles = c.n_styles
        c.aggregator.n_mels = c.n_mels
        c.decoder.hidden = c.hidden
        c.decoder.cond_dim = c.hidden
        c.decoder.n_mels = c.n_mels

        self.aggregator = ConditionAggregator(c.aggregator)
        self.whisper_branch = WhisperAwareConditioning(hidden=c.hidden, whisper_id=c.whisper_id)
        self.decoder = DiffusionDecoder(c.decoder)
        if c.schedule_type == "flow_matching":
            self.schedule = FlowMatchingSchedule(num_steps=c.diffusion_steps)
            self.sampler = FlowMatchingSampler(self.schedule)
            self._default_infer_steps = 4
        elif c.schedule_type == "diffusion":
            self.schedule = NoiseSchedule(num_steps=c.diffusion_steps)
            self.sampler = DDIMSampler(self.schedule)
            self._default_infer_steps = 50
        else:
            raise ValueError(f"Unknown schedule_type: {c.schedule_type!r}")
        self.vocoder = NSFHifiGANWrapper(
            ckpt_path=c.vocoder_ckpt, n_mels=c.n_mels, hop=c.hop, freeze=True
        )

        # Optional EMA hook. Set externally (Trainer or ckpt loader) so that
        # infer() uses the EMA-averaged weights instead of the raw training-
        # step weights. Without this assignment, EMA shadow values would
        # never reach inference.
        self.ema = None

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def build_cond(
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
        base = self.aggregator(
            content=content,
            f0=f0,
            uv=uv,
            loudness=loudness,
            style_id=style_id,
            ref_mel=ref_mel,
            speaker_id=speaker_id,
            style_vec=style_vec,
        )
        return self.whisper_branch(base, uv=uv, style_id=style_id)

    # ------------------------------------------------------------------
    # Training step (returns dict of losses; trainer aggregates)
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        mel = batch["mel"]
        B = mel.shape[0]
        cond = self.build_cond(
            content=batch["content"],
            f0=batch["f0"],
            uv=batch["uv"],
            loudness=batch["loudness"],
            style_id=batch["style_id"],
            ref_mel=batch.get("ref_mel"),
            speaker_id=batch.get("speaker_id"),
        )

        t = self.schedule.sample_random_t(B, device=mel.device)
        x_t, _, v_target = self.schedule.add_noise(mel, t)
        v_pred = self.decoder(x_t, t, cond)

        if "mask" in batch and batch["mask"] is not None:
            mask = batch["mask"].unsqueeze(1).float()  # (B, 1, T)
            diff = (v_pred - v_target) * mask
            denom = mask.sum().clamp(min=1.0) * v_pred.shape[1]
            loss = diff.pow(2).sum() / denom
        else:
            loss = (v_pred - v_target).pow(2).mean()

        return {"diffusion_loss": loss, "total_loss": loss}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        content: Tensor,
        f0: Tensor,
        uv: Tensor,
        loudness: Tensor,
        style_id: Tensor,
        ref_mel: Tensor | None = None,
        speaker_id: Tensor | None = None,
        style_vec: Tensor | None = None,
        num_steps: int | None = None,
        use_ema: bool = True,
    ) -> dict[str, Tensor]:
        if num_steps is None:
            num_steps = self._default_infer_steps

        # When EMA shadow weights have been attached (by the Trainer or by a
        # checkpoint loader), use those for inference instead of the live
        # training-step parameters. This is the half of EMA that actually
        # matters: without it the shadow values are accumulated but never
        # read.
        if use_ema and self.ema is not None:
            ema_ctx = self.ema.swap_in(self)
        else:
            ema_ctx = nullcontext()

        with ema_ctx:
            cond = self.build_cond(
                content=content,
                f0=f0,
                uv=uv,
                loudness=loudness,
                style_id=style_id,
                ref_mel=ref_mel,
                speaker_id=speaker_id,
                style_vec=style_vec,
            )
            B, _, T = cond.shape
            mel = self.sampler.sample(
                self.decoder, cond, shape=(B, self.cfg.n_mels, T), num_steps=num_steps
            )
            wav = self.vocoder(mel, f0)
        return {"mel": mel, "wav": wav}
