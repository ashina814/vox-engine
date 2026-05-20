import json

import numpy as np

from vox.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner


def _sin(sr=44_100, dur=1.0, freq=440.0, amp=0.3):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_eval_set(n=3):
    return [{"chunk_id": f"c{i:02d}", "style_id": i % 3} for i in range(n)]


def _identity_render_fn(item):
    """Returns identical ref/pred so metrics should be ~0."""
    wav = _sin()
    T = 100
    f0 = np.full(T, 220.0, dtype=np.float32)
    uv = np.ones(T, dtype=bool)
    return wav, wav, f0, f0, uv, uv, item["style_id"]


def _degraded_render_fn(item):
    """Different freq + 50% uv mismatch → non-zero metrics for every component."""
    ref = _sin(freq=440.0)
    pred = _sin(freq=660.0)
    T = 100
    f0_ref = np.full(T, 220.0, dtype=np.float32)
    f0_pred = np.full(T, 330.0, dtype=np.float32)
    uv_ref = np.ones(T, dtype=bool)
    uv_pred = np.ones(T, dtype=bool)
    uv_pred[:50] = False  # 50% disagreement; remaining 50 frames feed the F0 RMSE
    return ref, pred, f0_ref, f0_pred, uv_ref, uv_pred, item["style_id"]


def test_benchmark_identity_metrics_near_zero():
    runner = BenchmarkRunner(BenchmarkConfig(max_samples=2))
    res = runner.run(_make_eval_set(2), _identity_render_fn)
    assert res.metrics["mcd_db"] < 1e-3
    assert res.metrics["log_f0_rmse"] == 0.0
    assert res.metrics["uv_error"] == 0.0
    assert res.metrics["n_samples"] == 2


def test_benchmark_degraded_metrics_positive():
    runner = BenchmarkRunner(BenchmarkConfig(max_samples=2))
    res = runner.run(_make_eval_set(2), _degraded_render_fn)
    assert res.metrics["mcd_db"] > 1.0
    assert res.metrics["log_f0_rmse"] > 0.1
    assert abs(res.metrics["uv_error"] - 0.5) < 1e-9


def test_benchmark_includes_per_sample():
    runner = BenchmarkRunner(BenchmarkConfig(max_samples=3))
    res = runner.run(_make_eval_set(3), _identity_render_fn)
    assert len(res.per_sample) == 3
    for entry in res.per_sample:
        assert {"chunk_id", "mcd_db", "log_f0_rmse", "uv_error", "style_id"} <= set(entry)


def test_benchmark_json_roundtrip(tmp_path):
    runner = BenchmarkRunner(BenchmarkConfig(max_samples=2))
    res = runner.run(_make_eval_set(2), _identity_render_fn)
    out = tmp_path / "result.json"
    res.to_json(out)
    loaded = json.loads(out.read_text())
    assert "metrics" in loaded and "per_sample" in loaded
    assert loaded["metrics"]["n_samples"] == 2


def test_benchmark_style_separability_when_classifier_provided():
    def classifier(wav, sr):
        # Always returns class 0 — accuracy = (#samples with style_id==0) / n
        out = np.zeros(3)
        out[0] = 1.0
        return out

    runner = BenchmarkRunner(BenchmarkConfig(max_samples=3, classifier=classifier))
    res = runner.run(_make_eval_set(3), _identity_render_fn)
    assert "style_separability" in res.metrics
    # eval set styles: [0, 1, 2] → classifier always says 0 → acc = 1/3
    assert abs(res.metrics["style_separability"] - 1.0 / 3.0) < 1e-6
