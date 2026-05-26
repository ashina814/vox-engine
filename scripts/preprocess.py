"""Hydra-driven preprocess CLI.

Usage:
    uv run python scripts/preprocess.py data=opensinger
    uv run python scripts/preprocess.py data=synthetic preprocess.extract_content=false
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from vox.data.features.f0 import F0Extractor
from vox.data.preprocessing import PreprocessConfig, run_preprocessing
from vox.data.qa import QAConfig


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    qa_cfg = QAConfig(**OmegaConf.to_container(cfg.qa, resolve=True))  # type: ignore[arg-type]
    pp_cfg = PreprocessConfig(
        raw_dir=Path(cfg.data.raw_dir),
        processed_dir=Path(cfg.data.processed_dir),
        quarantine_dir=Path(cfg.data.quarantine_dir),
        index_path=Path(cfg.data.index_path),
        sr=cfg.audio.sr,
        hop=cfg.audio.hop,
        win=cfg.audio.win,
        min_s=cfg.chunking.min_s,
        max_s=cfg.chunking.max_s,
        silence_db=cfg.chunking.silence_db,
        min_silence_s=cfg.chunking.min_silence_s,
        extract_content=bool(cfg.get("preprocess", {}).get("extract_content", True)),
        qa=qa_cfg,
    )

    # Build style→id map. Caller can override via cfg.data.style_ids.
    if "style_ids" in cfg.data and cfg.data.style_ids is not None:
        style_to_id = dict(cfg.data.style_ids)
    else:
        style_to_id = {
            d.name: i for i, d in enumerate(sorted(pp_cfg.raw_dir.iterdir())) if d.is_dir()
        }

    f0 = F0Extractor(backend="torchcrepe", hop=pp_cfg.hop, sr=pp_cfg.sr)

    content_fn = None
    if pp_cfg.extract_content:
        from vox.data.features.content import ContentVecExtractor

        cv = ContentVecExtractor()
        content_fn = lambda wav, sr: cv(wav, src_sr=sr)  # noqa: E731

    df = run_preprocessing(
        pp_cfg,
        style_to_id=style_to_id,
        f0_fn=f0,
        content_fn=content_fn,
    )
    print(f"Processed {len(df)} chunks, quarantined {df.attrs.get('quarantined', 0)}")
    print(f"Index written: {pp_cfg.index_path}")


if __name__ == "__main__":
    main()
