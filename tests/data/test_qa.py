import numpy as np
import torch
import pytest

from vox.data.qa import (
    ChunkFeatures,
    ChunkQA,
    QAConfig,
    check_clip_ratio,
    check_duration,
    check_energy_dynamic_range,
    check_f0_continuity,
    check_f0_range,
    check_silence_ratio,
)


def test_f0_continuity_clean():
    f0 = np.full(100, 440.0, dtype=np.float32)
    passed, ratio = check_f0_continuity(f0)
    assert passed
    assert ratio == 0.0


def test_f0_continuity_detects_octave_jumps():
    f0 = np.array([220.0, 440.0] * 50, dtype=np.float32)  # constant 1-octave jumps
    passed, ratio = check_f0_continuity(f0)
    assert not passed
    assert ratio > 0.5


def test_f0_continuity_ignores_unvoiced():
    f0 = np.zeros(100, dtype=np.float32)
    passed, ratio = check_f0_continuity(f0)
    assert passed
    assert ratio == 0.0


def test_f0_range_in_bounds():
    f0 = np.full(50, 300.0, dtype=np.float32)
    uv = np.ones(50, dtype=bool)
    assert check_f0_range(f0, uv)[0]


def test_f0_range_out_of_bounds():
    f0 = np.full(50, 1500.0, dtype=np.float32)
    uv = np.ones(50, dtype=bool)
    assert not check_f0_range(f0, uv)[0]


def test_energy_flat_signal_fails():
    loud = np.full(100, 0.5, dtype=np.float32)
    assert not check_energy_dynamic_range(loud)[0]


def test_energy_dynamic_passes():
    rng = np.random.default_rng(0)
    loud = rng.uniform(0.0, 1.0, size=100).astype(np.float32)
    assert check_energy_dynamic_range(loud)[0]


def test_silence_ratio_mostly_voiced():
    uv = np.ones(100, dtype=bool)
    uv[:10] = False
    assert check_silence_ratio(uv)[0]


def test_silence_ratio_mostly_silent():
    uv = np.zeros(100, dtype=bool)
    uv[:10] = True
    assert not check_silence_ratio(uv)[0]


def test_duration_bounds():
    assert check_duration(5.0)[0]
    assert not check_duration(0.5)[0]
    assert not check_duration(20.0)[0]


def test_clip_ratio():
    rng = np.random.default_rng(0)
    clean = 0.1 * rng.standard_normal(10_000).astype(np.float32)
    assert check_clip_ratio(clean)[0]
    clipped = clean.copy()
    clipped[:1000] = 1.0  # 10% clipped
    assert not check_clip_ratio(clipped)[0]


def _good_chunk(sr=44_100) -> ChunkFeatures:
    rng = np.random.default_rng(0)
    n_frames = 100
    t = np.linspace(0, 5.0, 5 * sr, endpoint=False, dtype=np.float32)
    wav = torch.from_numpy(0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32))
    f0 = torch.full((n_frames,), 440.0)
    uv = torch.ones(n_frames, dtype=torch.bool)
    loud = torch.from_numpy(rng.uniform(0.1, 0.9, size=n_frames).astype(np.float32))
    return ChunkFeatures(wav=wav, sr=sr, f0=f0, uv=uv, loudness=loud)


def test_chunkqa_accepts_good_chunk():
    res = ChunkQA()(_good_chunk())
    assert res.passed, res.checks


def test_chunkqa_rejects_short_chunk():
    sr = 44_100
    short = ChunkFeatures(
        wav=torch.zeros(int(sr * 0.5)),
        sr=sr,
        f0=torch.full((10,), 440.0),
        uv=torch.ones(10, dtype=torch.bool),
        loudness=torch.linspace(0.1, 0.9, 10),
    )
    res = ChunkQA()(short)
    assert not res.passed
    assert "duration" in (res.reason or "")
