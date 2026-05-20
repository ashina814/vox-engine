"""End-to-end inference: WAV → features → conditioning → diffusion → vocoder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from vox.data.features.loudness import a_weighted_rms
from vox.data.features.mel import MelExtractor
from vox.data.features.uv import compute_uv
from vox.data.preprocessing import _load_wav
from vox.inference.autotune import get_scale, snap_to_scale
from vox.inference.style_blend import slerp_barycentric
from vox.models.vox_model import VoxModel


@dataclass
class InferenceRequest:
    input_wav: np.ndarray | Path | str
    sr: int = 44_100
    target_key: str | None = "C"
    target_mode: Literal["major", "minor"] = "major"
    style_weights: tuple[float, ...] = (1.0, 0.0, 0.0)
    autotune_strength: float = 0.8
    num_diffusion_steps: int = 50
    ref_mel: np.ndarray | None = None  # optional GST reference (n_mels, T_ref)


@dataclass
class InferenceResult:
    output_wav: np.ndarray  # float32, mono, sr=44_100
    sr: int = 44_100
    metadata: dict = field(default_factory=dict)


class InferencePipeline:
    """Stateless pipeline that consumes a request and returns an audio result.

    The pipeline holds references to a VoxModel and (optionally) external
    feature extractors. The F0 / Content extractors are passed as callables so
    test fixtures can substitute lightweight stand-ins without instantiating
    torchcrepe / ContentVec.
    """

    def __init__(
        self,
        model: VoxModel,
        f0_fn,
        content_fn,
        mel_ext: MelExtractor | None = None,
        sr: int = 44_100,
        hop: int = 512,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.f0_fn = f0_fn
        self.content_fn = content_fn
        self.mel_ext = mel_ext or MelExtractor(sr=sr, hop=hop)
        self.sr = sr
        self.hop = hop

    # ------------------------------------------------------------------

    def _load(self, src: np.ndarray | Path | str) -> Tensor:
        if isinstance(src, (str, Path)):
            return _load_wav(Path(src), self.sr)
        arr = np.asarray(src, dtype=np.float32).reshape(-1)
        return torch.from_numpy(arr)

    def _blend_style(self, weights: tuple[float, ...]) -> Tensor:
        emb = self.model.aggregator.style_emb.weight  # (n_styles, H)
        if len(weights) > emb.shape[0]:
            raise ValueError(
                f"Got {len(weights)} style weights but model has {emb.shape[0]} styles"
            )
        vecs = [emb[i].detach() for i in range(len(weights))]
        return slerp_barycentric(vecs, list(weights)).to(self.device).unsqueeze(0)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, req: InferenceRequest) -> InferenceResult:
        t0 = time.perf_counter()

        wav = self._load(req.input_wav).to(self.device)
        mel = self.mel_ext(wav).to(self.device)
        f0_t, _conf = self.f0_fn(wav)
        f0_np = f0_t.detach().cpu().numpy().astype(np.float32)

        if req.target_key is not None and req.autotune_strength > 0:
            f0_np = snap_to_scale(
                f0_np,
                get_scale(req.target_key, req.target_mode),
                strength=req.autotune_strength,
                sr_hop=self.hop,
                sr=self.sr,
            )
        f0_t = torch.from_numpy(f0_np).to(self.device)

        loudness = a_weighted_rms(wav.cpu(), hop=self.hop, sr=self.sr).to(self.device)
        uv = compute_uv(f0_t, loudness)
        content = self.content_fn(wav, self.sr).to(self.device)

        # Time-align everything to mel frame count.
        T = mel.shape[-1]
        f0_t = self._fit_T(f0_t, T)
        loudness = self._fit_T(loudness, T)
        uv = self._fit_T(uv.float(), T) > 0.5
        if content.shape[-1] != T:
            content = torch.nn.functional.interpolate(
                content.unsqueeze(0), size=T, mode="linear", align_corners=False
            ).squeeze(0)

        style_vec = self._blend_style(tuple(req.style_weights))

        ref_mel = None
        if req.ref_mel is not None:
            ref_mel = torch.from_numpy(req.ref_mel.astype(np.float32)).unsqueeze(0).to(self.device)

        out = self.model.infer(
            content=content.unsqueeze(0),
            f0=f0_t.unsqueeze(0),
            uv=uv.unsqueeze(0),
            loudness=loudness.unsqueeze(0),
            style_id=torch.zeros(1, dtype=torch.long, device=self.device),
            style_vec=style_vec,
            ref_mel=ref_mel,
            num_steps=req.num_diffusion_steps,
        )

        wav_out = out["wav"].squeeze(0).detach().cpu().numpy().astype(np.float32)

        elapsed = time.perf_counter() - t0
        return InferenceResult(
            output_wav=wav_out,
            sr=self.sr,
            metadata={
                "elapsed_s": elapsed,
                "rtf": elapsed / max(len(wav_out) / self.sr, 1e-6),
                "num_diffusion_steps": req.num_diffusion_steps,
                "style_weights": list(req.style_weights),
                "autotune_applied": req.target_key is not None,
                "T_mel": T,
            },
        )

    @staticmethod
    def _fit_T(x: Tensor, T: int) -> Tensor:
        if x.shape[-1] == T:
            return x
        # Linear interpolate 1-D tensor to length T.
        return torch.nn.functional.interpolate(
            x.view(1, 1, -1).float(), size=T, mode="linear", align_corners=False
        ).view(-1)
