"""End-to-end smoke test for the preprocessing loop with mocked extractors."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from vox.data.preprocessing import PreprocessConfig, run_preprocessing


def _fake_f0(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pretend torchcrepe: constant 440 Hz F0 at mel-aligned grid."""
    T = wav.shape[-1] // 512 + 1
    return torch.full((T,), 440.0), torch.full((T,), 0.9)


def _fake_content(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Pretend ContentVec: random (768, T_c) features."""
    T = wav.shape[-1] // 512 + 1
    return torch.randn(768, T)


def _write_tone_wav(path, duration_s: float, sr: int = 44_100, freq: float = 440.0):
    """Tone with a slow amplitude envelope so QA's energy-dynamics check passes."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    envelope = 0.3 + 0.4 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz amp modulation
    wav = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, wav, sr)


def test_preprocessing_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    (raw / "normal").mkdir(parents=True)
    (raw / "whisper").mkdir(parents=True)
    _write_tone_wav(raw / "normal" / "take01.wav", duration_s=8.0, freq=440.0)
    _write_tone_wav(raw / "whisper" / "take01.wav", duration_s=8.0, freq=330.0)

    cfg = PreprocessConfig(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        quarantine_dir=tmp_path / "quarantine",
        index_path=tmp_path / "index.parquet",
        val_ratio=0.0,  # deterministic: everything to train
    )
    df = run_preprocessing(
        cfg,
        style_to_id={"normal": 0, "whisper": 1},
        f0_fn=_fake_f0,
        content_fn=_fake_content,
    )

    assert len(df) == 2
    assert set(df["style"]) == {"normal", "whisper"}
    assert (cfg.index_path).exists()

    # Each output file must exist and load as the expected shape.
    for _, row in df.iterrows():
        mel = np.load(row["mel_path"])
        f0 = np.load(row["f0_path"])
        uv = np.load(row["uv_path"])
        content = np.load(row["content_path"])
        loud = np.load(row["loudness_path"])
        assert mel.shape[0] == 128
        assert f0.shape == uv.shape == loud.shape
        assert content.shape[0] == 768


def test_preprocessing_quarantines_silent_chunks(tmp_path):
    """A silent input produces no chunks (chunking filter) → empty index."""
    raw = tmp_path / "raw"
    (raw / "normal").mkdir(parents=True)
    silent = np.zeros(8 * 44_100, dtype=np.float32)
    sf.write(raw / "normal" / "silent.wav", silent, 44_100)

    cfg = PreprocessConfig(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        quarantine_dir=tmp_path / "quarantine",
        index_path=tmp_path / "index.parquet",
    )
    df = run_preprocessing(cfg, style_to_id={"normal": 0}, f0_fn=_fake_f0, content_fn=None)
    assert len(df) == 0


def test_preprocessing_skips_content_when_no_fn(tmp_path):
    raw = tmp_path / "raw"
    (raw / "normal").mkdir(parents=True)
    _write_tone_wav(raw / "normal" / "take01.wav", duration_s=8.0)

    cfg = PreprocessConfig(
        raw_dir=raw,
        processed_dir=tmp_path / "processed",
        quarantine_dir=tmp_path / "quarantine",
        index_path=tmp_path / "index.parquet",
        val_ratio=0.0,
        extract_content=False,
    )
    df = run_preprocessing(cfg, style_to_id={"normal": 0}, f0_fn=_fake_f0, content_fn=None)
    assert len(df) == 1
    assert df.iloc[0]["content_path"] == ""
