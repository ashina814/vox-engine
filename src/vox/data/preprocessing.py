"""Core preprocessing loop, decoupled from Hydra/CLI for testability.

Layout expected by ``run_preprocessing``:

    raw_dir/
      {style}/{take}.wav

Produces:

    processed_dir/
      mel/{style}/{chunk_id}.npy
      f0/{style}/{chunk_id}.npy
      uv/{style}/{chunk_id}.npy
      content/{style}/{chunk_id}.npy        # only if extract_content=True
      loudness/{style}/{chunk_id}.npy
    quarantine_dir/
      {chunk_id}.json
    index.parquet                            # rows for QA-passed chunks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor

from vox.data.chunking import Chunk, chunk_wav
from vox.data.features.loudness import a_weighted_rms
from vox.data.features.mel import MelExtractor
from vox.data.features.uv import compute_uv
from vox.data.qa import ChunkFeatures, ChunkQA, QAConfig


@dataclass
class PreprocessConfig:
    raw_dir: Path
    processed_dir: Path
    quarantine_dir: Path
    index_path: Path
    sr: int = 44_100
    hop: int = 512
    win: int = 2048
    min_s: float = 5.0
    max_s: float = 10.0
    silence_db: float = -40.0
    min_silence_s: float = 0.3
    val_ratio: float = 0.05
    extract_content: bool = True
    qa: QAConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.quarantine_dir = Path(self.quarantine_dir)
        self.index_path = Path(self.index_path)
        if self.qa is None:
            self.qa = QAConfig()


# A minimal extractor protocol so tests can inject fakes.
F0Fn = Callable[[Tensor], tuple[Tensor, Tensor]]
ContentFn = Callable[[Tensor, int], Tensor]


def _iter_raw_wavs(raw_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (style_name, wav_path) for every wav under raw_dir/{style}/."""
    for style_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for wav_path in sorted(style_dir.glob("*.wav")):
            yield style_dir.name, wav_path


def _load_wav(path: Path, target_sr: int) -> Tensor:
    wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav_np = wav_np.mean(axis=1)  # mono
    wav = torch.from_numpy(wav_np)
    if sr != target_sr:
        import torchaudio.functional as F_audio

        wav = F_audio.resample(wav, sr, target_sr)
    return wav.float()


def _save_chunk_features(
    out_dir: Path,
    style: str,
    chunk_id: str,
    feats: dict[str, np.ndarray],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, arr in feats.items():
        d = out_dir / name / style
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{chunk_id}.npy"
        np.save(p, arr)
        paths[f"{name}_path"] = str(p)
    return paths


def _quarantine(quarantine_dir: Path, chunk_id: str, reason: str, metrics: dict) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    payload = {"chunk_id": chunk_id, "reason": reason, "metrics": metrics}
    (quarantine_dir / f"{chunk_id}.json").write_text(json.dumps(payload, indent=2))


def run_preprocessing(
    cfg: PreprocessConfig,
    style_to_id: dict[str, int],
    f0_fn: F0Fn,
    content_fn: ContentFn | None = None,
    val_rng_seed: int = 0,
) -> pd.DataFrame:
    """Process every wav under cfg.raw_dir and write features + index.

    ``f0_fn(wav) -> (f0, confidence)`` and ``content_fn(wav, sr) -> (768, T_c)``
    are injected so tests can pass synthetic stand-ins instead of downloading
    real models.
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.quarantine_dir.mkdir(parents=True, exist_ok=True)

    mel_ext = MelExtractor(sr=cfg.sr, hop=cfg.hop, win=cfg.win)
    qa = ChunkQA(cfg.qa)
    rng = np.random.default_rng(val_rng_seed)

    rows: list[dict] = []
    quarantined = 0

    for style, wav_path in _iter_raw_wavs(cfg.raw_dir):
        if style not in style_to_id:
            continue
        wav = _load_wav(wav_path, cfg.sr)
        chunks: list[Chunk] = chunk_wav(
            wav,
            sr=cfg.sr,
            min_s=cfg.min_s,
            max_s=cfg.max_s,
            silence_db=cfg.silence_db,
            min_silence_s=cfg.min_silence_s,
        )

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{wav_path.stem}_{idx:04d}"

            mel = mel_ext(chunk.wav)  # (n_mels, T_mel)
            f0, confidence = f0_fn(chunk.wav)
            loudness = a_weighted_rms(chunk.wav, hop=cfg.hop, win=cfg.win, sr=cfg.sr)
            uv = compute_uv(f0, loudness)

            feats_np = {
                "mel": mel.numpy(),
                "f0": f0.numpy(),
                "uv": uv.numpy().astype(np.bool_),
                "loudness": loudness.numpy(),
            }
            if content_fn is not None:
                content = content_fn(chunk.wav, cfg.sr)
                feats_np["content"] = content.numpy()

            result = qa(ChunkFeatures(wav=chunk.wav, sr=cfg.sr, f0=f0, uv=uv, loudness=loudness))
            if not result.passed:
                _quarantine(
                    cfg.quarantine_dir,
                    chunk_id,
                    result.reason or "unknown",
                    result.metrics,
                )
                quarantined += 1
                continue

            paths = _save_chunk_features(cfg.processed_dir, style, chunk_id, feats_np)
            split = "val" if rng.random() < cfg.val_ratio else "train"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "style_id": style_to_id[style],
                    "style": style,
                    "split": split,
                    "duration_s": chunk.duration_s,
                    "source_wav": str(wav_path),
                    **paths,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Ensure every chunk has a content_path column for downstream Dataset,
        # even if content extraction was skipped.
        if "content_path" not in df.columns:
            df["content_path"] = ""
        cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cfg.index_path)
    df.attrs["quarantined"] = quarantined
    return df
