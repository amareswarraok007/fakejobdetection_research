"""Detection metrics (PRD Section 6.5). Positive class = 1 = fake.

- Threshold-free: PR-AUC (average precision) and ROC-AUC, from the scores.
- At a threshold t (a posting is called fake when score >= t): precision, recall,
  fake-class F1, macro-F1, accuracy and the confusion matrix counts.
Bootstrap confidence intervals and McNemar's test are in ``stats.py``.
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cvx.data.split import SplitIndices

RATE_METRICS = ("precision", "recall", "f1", "macro_f1", "accuracy")
SCORE_METRICS = ("pr_auc", "roc_auc")


def confusion_counts(y: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    """TN, FP, FN, TP for 0/1 labels and 0/1 predictions."""
    y, predicted = np.asarray(y).astype(int), np.asarray(predicted).astype(int)
    return {
        "tn": int(np.sum((y == 0) & (predicted == 0))),
        "fp": int(np.sum((y == 0) & (predicted == 1))),
        "fn": int(np.sum((y == 1) & (predicted == 0))),
        "tp": int(np.sum((y == 1) & (predicted == 1))),
    }


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    return _ratio(2 * precision * recall, precision + recall)


def rates_from_counts(c: dict[str, int]) -> dict[str, float]:
    """Precision, recall, fake-class F1, macro-F1 and accuracy from confusion counts."""
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    recall = _ratio(c["tp"], c["tp"] + c["fn"])
    # Same formulas for the genuine class (class 0) give the other half of macro-F1.
    f1_genuine = _f1(_ratio(c["tn"], c["tn"] + c["fn"]), _ratio(c["tn"], c["tn"] + c["fp"]))
    f1_fake = _f1(precision, recall)
    total = c["tn"] + c["fp"] + c["fn"] + c["tp"]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_fake,
        "macro_f1": (f1_fake + f1_genuine) / 2,
        "accuracy": _ratio(c["tp"] + c["tn"], total),
    }


def at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """All threshold metrics plus confusion counts when ``score >= threshold`` means fake."""
    counts = confusion_counts(y, np.asarray(scores) >= threshold)
    return {"threshold": float(threshold), **rates_from_counts(counts), **counts}


def threshold_free(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """PR-AUC (average precision) and ROC-AUC; None if only one class is present."""
    if len(np.unique(y)) < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def evaluate_scores(
    y: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    """Threshold-free metrics and metrics at every named threshold."""
    return {
        "threshold_free": threshold_free(y, scores),
        "at_threshold": {name: at_threshold(y, scores, t) for name, t in thresholds.items()},
    }


def split_summary(y: np.ndarray, indices: SplitIndices) -> dict[str, dict[str, Any]]:
    """Rows, fake count and fake share in each split."""
    summary = {}
    for name in ("train", "val", "test"):
        labels = np.asarray(y)[getattr(indices, name)]
        fake = int(labels.sum())
        summary[name] = {
            "rows": int(len(labels)),
            "fake": fake,
            "fake_share": _ratio(fake, len(labels)),
        }
    return summary
