"""Aggregate evaluator: model + eval dataset → metric dict.

Each chunk in the eval set is passed through the model's inference path; the
resulting wav is compared against the ground-truth audio to compute MCD, F0
RMSE, and UV error. Style separability is optional (requires a classifier).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from vox.evaluation.f0_rmse import f0_rmse
from vox.evaluation.mcd import mel_cepstral_distortion
from vox.evaluation.style_separability import StyleClassifier, StyleSeparabilityEvaluator
from vox.evaluation.uv_error import uv_error_rate


@dataclass
class BenchmarkConfig:
    sr: int = 44_100
    hop: int = 512
    n_mcep: int = 13
    max_samples: int | None = None  # cap eval iterations (None = full)
    n_styles: int = 3
    classifier: StyleClassifier | None = None


@dataclass
class BenchmarkResult:
    metrics: dict[str, float]
    per_sample: list[dict]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"metrics": self.metrics, "per_sample": self.per_sample},
            indent=2,
        ))


# Audio-render callable: turn an eval item into (ref_wav, pred_wav, f0_ref, f0_pred, uv_ref, uv_pred, style_id).
# We inject this so the runner stays decoupled from the heavy model.
RenderFn = Callable[[dict], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]


class BenchmarkRunner:
    def __init__(self, cfg: BenchmarkConfig | None = None) -> None:
        self.cfg = cfg or BenchmarkConfig()

    def run(self, eval_set: Iterable[dict], render_fn: RenderFn) -> BenchmarkResult:
        per_sample: list[dict] = []
        samples_by_style: dict[int, list[np.ndarray]] = {}

        start = time.perf_counter()
        for i, item in enumerate(eval_set):
            if self.cfg.max_samples is not None and i >= self.cfg.max_samples:
                break
            ref_wav, pred_wav, f0_ref, f0_pred, uv_ref, uv_pred, style_id = render_fn(item)
            mcd_db = mel_cepstral_distortion(
                ref_wav, pred_wav, sr=self.cfg.sr, n_mcep=self.cfg.n_mcep, hop=self.cfg.hop
            )
            f0r = f0_rmse(f0_ref, f0_pred, uv=uv_ref & uv_pred, log=True)
            uv_err = uv_error_rate(uv_ref, uv_pred)
            per_sample.append({
                "chunk_id": item.get("chunk_id", f"sample_{i:04d}"),
                "mcd_db": mcd_db,
                "log_f0_rmse": f0r,
                "uv_error": uv_err,
                "style_id": int(style_id),
            })
            samples_by_style.setdefault(int(style_id), []).append(pred_wav)

        elapsed = time.perf_counter() - start

        metrics: dict[str, float] = {}
        if per_sample:
            metrics["mcd_db"] = float(np.mean([s["mcd_db"] for s in per_sample]))
            metrics["log_f0_rmse"] = float(np.mean([s["log_f0_rmse"] for s in per_sample]))
            metrics["uv_error"] = float(np.mean([s["uv_error"] for s in per_sample]))
        metrics["n_samples"] = float(len(per_sample))
        metrics["elapsed_s"] = float(elapsed)

        if self.cfg.classifier is not None and samples_by_style:
            ev = StyleSeparabilityEvaluator(
                self.cfg.classifier, n_classes=self.cfg.n_styles, sr=self.cfg.sr
            )
            metrics["style_separability"] = float(ev(samples_by_style))

        return BenchmarkResult(metrics=metrics, per_sample=per_sample)


# Convenience: build a render_fn that consumes Dataset rows + a VoxModel.
def make_default_render_fn(model, pipeline_fn) -> RenderFn:
    """Wrap a VoxModel-style inference callable into the RenderFn shape."""

    def render(item: dict) -> tuple:
        return pipeline_fn(model, item)

    return render
