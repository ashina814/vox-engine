"""Silence-aware chunking of long audio into 5-10s segments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Chunk:
    """A single audio chunk with sample-level provenance."""

    wav: Tensor  # (T_wav,) float32
    sr: int
    start_sample: int  # inclusive offset into the source wav
    end_sample: int  # exclusive

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / self.sr

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample


def chunk_wav(
    wav: Tensor,
    sr: int,
    min_s: float = 5.0,
    max_s: float = 10.0,
    silence_db: float = -40.0,
    min_silence_s: float = 0.3,
) -> list[Chunk]:
    """Split a long waveform into chunks of [min_s, max_s] seconds.

    Strategy:
      1. Detect non-silent intervals via librosa.effects.split (top_db = -silence_db).
      2. Greedily merge consecutive non-silent intervals until reaching max_s.
      3. If a merged span exceeds max_s, hard-split it at max_s boundaries.
      4. Drop chunks shorter than min_s.

    Args:
        wav: (T_wav,) float32 audio tensor in [-1, 1].
        sr: sample rate.
        min_s: minimum chunk duration (seconds). Shorter chunks are discarded.
        max_s: maximum chunk duration. Spans longer are hard-split.
        silence_db: dBFS threshold; samples below are considered silent.
        min_silence_s: minimum silence run length to count as a boundary.

    Returns:
        list of Chunk in source order.
    """
    if wav.dim() != 1:
        raise ValueError(f"Expected 1-D wav, got shape {wav.shape}")
    if min_s <= 0 or max_s < min_s:
        raise ValueError(f"Invalid bounds: min_s={min_s}, max_s={max_s}")

    import librosa

    wav_np = wav.detach().cpu().numpy().astype(np.float32)

    # librosa.effects.split degenerates to the full span when the input is silent
    # (log10(0) reference). Short-circuit here so callers get an empty list.
    silence_amp = 10 ** (silence_db / 20.0)
    if float(np.abs(wav_np).max()) < silence_amp:
        return []

    top_db = -silence_db  # librosa convention: top_db is positive, dB below reference
    frame_length = max(1, int(min_silence_s * sr))

    intervals = librosa.effects.split(
        wav_np, top_db=top_db, frame_length=frame_length, hop_length=frame_length // 4
    )
    if len(intervals) == 0:
        return []

    min_samples = int(min_s * sr)
    max_samples = int(max_s * sr)

    # Greedy merge: extend current chunk while next non-silent interval still fits.
    chunks: list[tuple[int, int]] = []
    cur_start, cur_end = int(intervals[0][0]), int(intervals[0][1])
    for s, e in intervals[1:]:
        s, e = int(s), int(e)
        if e - cur_start <= max_samples:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    chunks.append((cur_start, cur_end))

    # Hard-split any chunk longer than max_samples.
    split_chunks: list[tuple[int, int]] = []
    for s, e in chunks:
        while e - s > max_samples:
            split_chunks.append((s, s + max_samples))
            s += max_samples
        split_chunks.append((s, e))

    # Drop too-short chunks.
    result: list[Chunk] = []
    for s, e in split_chunks:
        if e - s < min_samples:
            continue
        result.append(
            Chunk(
                wav=torch.from_numpy(wav_np[s:e]),
                sr=sr,
                start_sample=s,
                end_sample=e,
            )
        )
    return result
