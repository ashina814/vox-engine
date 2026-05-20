"""Index-driven dataset of preprocessed feature chunks.

The preprocessing CLI writes one row per chunk to ``index.parquet`` with paths
to each feature .npy. ``VoxDataset`` lazily loads them; ``collate_fn`` pads
variable-length feature tensors and produces a mel-frame mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

FEATURE_PATH_COLS = ("mel_path", "f0_path", "uv_path", "content_path", "loudness_path")


class VoxDataset(Dataset):
    """Reads index.parquet and lazy-loads .npy features per chunk.

    Expected columns in ``index.parquet``:
      chunk_id, style_id, split, duration_s,
      mel_path, f0_path, uv_path, content_path, loudness_path
    """

    def __init__(
        self,
        index_path: str | Path,
        split: Literal["train", "val"] | None = "train",
    ) -> None:
        self.index_path = Path(index_path)
        df = pd.read_parquet(self.index_path)
        if split is not None:
            if "split" not in df.columns:
                raise KeyError(f"index missing 'split' column: {self.index_path}")
            df = df[df["split"] == split].reset_index(drop=True)
        for col in FEATURE_PATH_COLS + ("chunk_id", "style_id"):
            if col not in df.columns:
                raise KeyError(f"index missing required column {col!r}")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        row = self.df.iloc[i]
        mel = torch.from_numpy(np.load(row["mel_path"])).float()
        f0 = torch.from_numpy(np.load(row["f0_path"])).float()
        uv = torch.from_numpy(np.load(row["uv_path"]))
        content = torch.from_numpy(np.load(row["content_path"])).float()
        loudness = torch.from_numpy(np.load(row["loudness_path"])).float()
        return {
            "mel": mel,  # (n_mels, T)
            "f0": f0,  # (T,)
            "uv": uv.to(torch.bool),  # (T,)
            "content": content,  # (768, T)
            "loudness": loudness,  # (T,)
            "style_id": torch.tensor(int(row["style_id"]), dtype=torch.long),
            "chunk_id": str(row["chunk_id"]),
        }


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length features to the batch max along the time axis.

    Returns a dict with stacked tensors plus ``mask`` of shape (B, T) where
    1 = valid frame, 0 = padded.
    """
    if not batch:
        raise ValueError("Empty batch")

    lengths = [item["mel"].shape[-1] for item in batch]
    T_max = max(lengths)
    B = len(batch)
    n_mels = batch[0]["mel"].shape[0]
    content_dim = batch[0]["content"].shape[0]

    mel = torch.zeros(B, n_mels, T_max, dtype=torch.float32)
    f0 = torch.zeros(B, T_max, dtype=torch.float32)
    uv = torch.zeros(B, T_max, dtype=torch.bool)
    content = torch.zeros(B, content_dim, T_max, dtype=torch.float32)
    loudness = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    style_id = torch.zeros(B, dtype=torch.long)
    chunk_ids: list[str] = []

    for i, item in enumerate(batch):
        T = item["mel"].shape[-1]
        mel[i, :, :T] = item["mel"]
        f0[i, :T] = item["f0"]
        uv[i, :T] = item["uv"]
        # Content may be stored at a different time resolution; truncate or pad to T.
        c = item["content"]
        c_t = c.shape[-1]
        copy_t = min(c_t, T)
        content[i, :, :copy_t] = c[:, :copy_t]
        loudness[i, :T] = item["loudness"]
        mask[i, :T] = True
        style_id[i] = item["style_id"]
        chunk_ids.append(item["chunk_id"])

    return {
        "mel": mel,
        "f0": f0,
        "uv": uv,
        "content": content,
        "loudness": loudness,
        "mask": mask,
        "style_id": style_id,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "chunk_ids": chunk_ids,
    }
