"""Style-separability evaluator.

Concept: given a bank of generated samples grouped by *intended* style id, a
classifier should be able to recover that style from the audio. This module
computes that classifier's top-1 accuracy as a scalar `style_separability`
metric. The design spec targets >80%.

For Phase A the real classifier (a SER model) is not yet trained. We expose a
clean callable interface so any classifier — even a placeholder — can be
plugged in via dependency injection. Phase B drops in a trained model.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Callable accepting (T,) float32 audio at sr and returning a vector of
# class logits or probabilities with shape (n_classes,).
StyleClassifier = Callable[[np.ndarray, int], np.ndarray]


class StyleSeparabilityEvaluator:
    """Score classifier accuracy on (style_id -> list of generated wavs).

    Args:
        classifier: a callable mapping ``(wav, sr) -> (n_classes,)`` logits.
        n_classes: number of style classes the classifier emits.
        sr: sample rate to pass to the classifier.
    """

    def __init__(self, classifier: StyleClassifier, n_classes: int, sr: int = 44_100) -> None:
        self.classifier = classifier
        self.n_classes = n_classes
        self.sr = sr

    def __call__(self, samples_by_style: dict[int, Sequence[np.ndarray]]) -> float:
        """Run the classifier on each (style_id, audio) pair, return accuracy."""
        total = 0
        correct = 0
        for true_id, audios in samples_by_style.items():
            if true_id < 0 or true_id >= self.n_classes:
                raise ValueError(f"style id {true_id} out of range [0, {self.n_classes})")
            for wav in audios:
                logits = np.asarray(self.classifier(np.asarray(wav, dtype=np.float32), self.sr))
                if logits.shape != (self.n_classes,):
                    raise ValueError(
                        f"classifier returned shape {logits.shape}, expected ({self.n_classes},)"
                    )
                pred = int(np.argmax(logits))
                correct += int(pred == true_id)
                total += 1
        if total == 0:
            return 0.0
        return correct / total
