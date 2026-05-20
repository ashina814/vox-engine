import numpy as np
import pytest

from vox.evaluation.style_separability import StyleSeparabilityEvaluator


def _energy_classifier(n_classes: int):
    """Picks a class based on RMS energy bucket — a trivial deterministic stand-in."""

    def fn(wav: np.ndarray, sr: int) -> np.ndarray:
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        idx = min(n_classes - 1, int(rms * 10) // 1)
        logits = np.zeros(n_classes)
        logits[idx] = 1.0
        return logits

    return fn


def test_perfect_classifier_accuracy_1():
    """Build samples whose RMS bucket equals the intended style id."""
    n_classes = 3
    samples = {
        0: [np.full(1000, 0.05, dtype=np.float32)],  # rms 0.05 → bucket 0
        1: [np.full(1000, 0.15, dtype=np.float32)],  # rms 0.15 → bucket 1
        2: [np.full(1000, 0.25, dtype=np.float32)],  # rms 0.25 → bucket 2
    }
    evaluator = StyleSeparabilityEvaluator(_energy_classifier(n_classes), n_classes)
    assert evaluator(samples) == 1.0


def test_misclassified_lowers_accuracy():
    n_classes = 3
    samples = {
        0: [np.full(1000, 0.05, dtype=np.float32)],
        1: [np.full(1000, 0.05, dtype=np.float32)],  # rms bucket 0, but labeled 1
    }
    evaluator = StyleSeparabilityEvaluator(_energy_classifier(n_classes), n_classes)
    assert evaluator(samples) == 0.5


def test_empty_input_returns_zero():
    evaluator = StyleSeparabilityEvaluator(_energy_classifier(3), n_classes=3)
    assert evaluator({}) == 0.0


def test_classifier_shape_mismatch_raises():
    def bad(wav, sr):
        return np.zeros(2)  # wrong size

    evaluator = StyleSeparabilityEvaluator(bad, n_classes=3)
    with pytest.raises(ValueError):
        evaluator({0: [np.zeros(100, dtype=np.float32)]})


def test_out_of_range_style_id_raises():
    evaluator = StyleSeparabilityEvaluator(_energy_classifier(3), n_classes=3)
    with pytest.raises(ValueError):
        evaluator({5: [np.zeros(100, dtype=np.float32)]})
