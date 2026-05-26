"""Hydra inference CLI.

Usage:
    uv run python scripts/infer.py input_wav=path/to.wav output_wav=out.wav
    uv run python scripts/infer.py input_wav=in.wav output_wav=out.wav \
        inference.target_key=D inference.num_diffusion_steps=20

Notes:
- ``ckpt_path`` (optional): VoxModel state_dict checkpoint. When omitted, an
  untrained model is used — useful for smoke-running the wiring.
- ``inference.skip_content``: bypass ContentVec download (uses random content
  features). Only meaningful for smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig

from vox.data.features.f0 import F0Extractor
from vox.inference.pipeline import InferencePipeline, InferenceRequest
from vox.models.vox_model import VoxModel, VoxModelConfig


def _build_content_fn(skip: bool):
    if skip:

        def fake(wav, sr):
            T = wav.shape[-1] // 512 + 1
            return torch.randn(768, T)

        return fake
    from vox.data.features.content import ContentVecExtractor

    cv = ContentVecExtractor()
    return lambda wav, sr: cv(wav, src_sr=sr)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    in_wav = cfg.get("input_wav")
    out_wav = cfg.get("output_wav", "vox_out.wav")
    if in_wav is None:
        raise SystemExit("Pass input_wav=path/to.wav on the command line.")

    model = VoxModel(VoxModelConfig())
    ckpt = cfg.get("ckpt_path")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded VoxModel weights from {ckpt}")

    f0 = F0Extractor(backend="torchcrepe", hop=cfg.audio.hop, sr=cfg.audio.sr)
    content_fn = _build_content_fn(skip=bool(cfg.get("inference", {}).get("skip_content", False)))

    pipeline = InferencePipeline(
        model=model,
        f0_fn=f0,
        content_fn=content_fn,
        sr=cfg.audio.sr,
        hop=cfg.audio.hop,
    )

    style_weights = cfg.get("inference", {}).get("style_weights", [1.0, 0.0, 0.0])
    req = InferenceRequest(
        input_wav=Path(in_wav),
        sr=cfg.audio.sr,
        target_key=cfg.inference.target_key,
        autotune_strength=cfg.inference.autotune_strength,
        num_diffusion_steps=cfg.inference.num_diffusion_steps,
        style_weights=tuple(style_weights),
    )
    res = pipeline(req)
    sf.write(out_wav, res.output_wav, res.sr)
    print(
        f"Wrote {out_wav}  ({len(res.output_wav)/res.sr:.2f} s)  " f"rtf={res.metadata['rtf']:.2f}"
    )


if __name__ == "__main__":
    main()
