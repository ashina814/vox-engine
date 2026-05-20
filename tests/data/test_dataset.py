import numpy as np
import pandas as pd
import pytest
import torch

from vox.data.dataset import VoxDataset, collate_fn


def _make_chunk(tmp_path, chunk_id: str, T: int, style_id: int = 0, split: str = "train"):
    mel = np.random.rand(128, T).astype(np.float32)
    f0 = np.random.rand(T).astype(np.float32) * 400 + 100
    uv = (np.random.rand(T) > 0.3).astype(np.bool_)
    content = np.random.rand(768, T).astype(np.float32)
    loudness = np.random.rand(T).astype(np.float32)
    paths = {}
    for name, arr in [
        ("mel", mel),
        ("f0", f0),
        ("uv", uv),
        ("content", content),
        ("loudness", loudness),
    ]:
        p = tmp_path / f"{chunk_id}_{name}.npy"
        np.save(p, arr)
        paths[f"{name}_path"] = str(p)
    return {
        "chunk_id": chunk_id,
        "style_id": style_id,
        "split": split,
        "duration_s": T * 512 / 44100.0,
        **paths,
    }


@pytest.fixture
def index_path(tmp_path):
    rows = [
        _make_chunk(tmp_path, "a", T=200, style_id=0, split="train"),
        _make_chunk(tmp_path, "b", T=250, style_id=1, split="train"),
        _make_chunk(tmp_path, "c", T=180, style_id=0, split="val"),
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "index.parquet"
    df.to_parquet(p)
    return p


def test_dataset_filters_split(index_path):
    train = VoxDataset(index_path, split="train")
    val = VoxDataset(index_path, split="val")
    assert len(train) == 2
    assert len(val) == 1


def test_dataset_returns_expected_keys(index_path):
    ds = VoxDataset(index_path, split="train")
    item = ds[0]
    expected = {"mel", "f0", "uv", "content", "loudness", "style_id", "chunk_id"}
    assert expected.issubset(item.keys())
    assert item["mel"].shape[0] == 128
    assert item["content"].shape[0] == 768


def test_collate_pads_to_batch_max(index_path):
    ds = VoxDataset(index_path, split="train")
    batch = collate_fn([ds[0], ds[1]])
    assert batch["mel"].shape == (2, 128, 250)
    assert batch["f0"].shape == (2, 250)
    assert batch["content"].shape == (2, 768, 250)
    assert batch["mask"].shape == (2, 250)
    # First chunk was T=200 → tail of mask must be zero
    assert batch["mask"][0, :200].all() and not batch["mask"][0, 200:].any()
    assert batch["mask"][1].all()
    assert batch["lengths"].tolist() == [200, 250]


def test_collate_padded_regions_are_zero(index_path):
    ds = VoxDataset(index_path, split="train")
    batch = collate_fn([ds[0], ds[1]])
    assert torch.all(batch["mel"][0, :, 200:] == 0)
    assert torch.all(batch["f0"][0, 200:] == 0)


def test_dataset_missing_split_column_raises(tmp_path):
    df = pd.DataFrame(
        [
            {
                "chunk_id": "x",
                "style_id": 0,
                "mel_path": "x",
                "f0_path": "x",
                "uv_path": "x",
                "content_path": "x",
                "loudness_path": "x",
            }
        ]
    )
    p = tmp_path / "index.parquet"
    df.to_parquet(p)
    with pytest.raises(KeyError):
        VoxDataset(p, split="train")
