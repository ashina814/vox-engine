"""Hydra entry that launches the gradio GUI.

Usage:
    uv run python scripts/ui.py
    uv run python scripts/ui.py ckpt_path=ckpts/step_00010000.pt ui.skip_content=true
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from vox.data.features.f0 import F0Extractor
from vox.inference.pipeline import InferencePipeline
from vox.models.vox_model import VoxModel, VoxModelConfig
from vox.ui.gradio_app import launch


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
    model = VoxModel(VoxModelConfig())
    ckpt = cfg.get("ckpt_path")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)

    f0 = F0Extractor(backend="torchcrepe", hop=cfg.audio.hop, sr=cfg.audio.sr)
    content_fn = _build_content_fn(skip=bool(cfg.get("ui", {}).get("skip_content", False)))
    pipeline = InferencePipeline(
        model=model,
        f0_fn=f0,
        content_fn=content_fn,
        sr=cfg.audio.sr,
        hop=cfg.audio.hop,
    )
    launch(
        pipeline,
        n_styles=int(cfg.get("ui", {}).get("n_styles", 3)),
        server_name=cfg.get("ui", {}).get("host", "127.0.0.1"),
        server_port=int(cfg.get("ui", {}).get("port", 7860)),
        inbrowser=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
