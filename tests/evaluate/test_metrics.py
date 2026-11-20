"""Tests for cvx.evaluate.metrics on a hand-calculated example.

Example (5 postings): labels   y = [1, 1, 0, 0, 0]
                      scores   s = [0.9, 0.4, 0.6, 0.2, 0.1]
At threshold 0.5 the predictions are [1, 0, 1, 0, 0]:
  TP = 1 (0.9), FN = 1 (0.4), FP = 1 (0.6), TN = 2
  precision = 1/2, recall = 1/2, F1 = 1/2, accuracy = 3/5
  genuine class: precision = TN/(TN+FN) = 2/3, recall = TN/(TN+FP) = 2/3, F1 = 2/3
  macro-F1 = (1/2 + 2/3) / 2 = 7/12
ROC-AUC: of the 2 x 3 fake/genuine pairs, the fake one scores higher in 5 -> 5/6.
PR-AUC (average precision): ranking 0.9(fake) 0.6(gen) 0.4(fake) ->
  recall steps of 1/2 at precision 1 and 2/3 -> 1/2 * 1 + 1/2 * 2/3 = 5/6.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cvx.data.split import SplitIndices
from cvx.evaluate.metrics import (
    at_threshold,
    confusion_counts,
    evaluate_scores,
    split_summary,
    threshold_free,
)

Y = np.array([1, 1, 0, 0, 0])
S = np.array([0.9, 0.4, 0.6, 0.2, 0.1])


def test_confusion_counts() -> None:
    assert confusion_counts(Y, S >= 0.5) == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}


def test_at_threshold_hand_values() -> None:
    m = at_threshold(Y, S, 0.5)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["f1"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["macro_f1"] == pytest.approx(7 / 12)
    assert m["threshold"] == 0.5


def test_at_threshold_matches_sklearn() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    m, pred = at_threshold(y, s, 0.3), (s >= 0.3).astype(int)
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))


def test_zero_division_gives_zero() -> None:
    m = at_threshold(Y, S, 0.95)  # nothing predicted fake
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_threshold_free_hand_values() -> None:
    result = threshold_free(Y, S)
    assert result["roc_auc"] == pytest.approx(5 / 6)
    assert result["pr_auc"] == pytest.approx(5 / 6)


def test_threshold_free_single_class() -> None:
    assert threshold_free(np.zeros(3), np.array([0.1, 0.2, 0.3])) == {
        "pr_auc": None,
        "roc_auc": None,
    }


def test_evaluate_scores_structure() -> None:
    result = evaluate_scores(Y, S, {"0.5": 0.5, "t_f1": 0.4})
    assert set(result) == {"threshold_free", "at_threshold"}
    assert result["at_threshold"]["t_f1"]["recall"] == 1.0  # 0.9 and 0.4 both >= 0.4


def test_split_summary() -> None:
    split = SplitIndices(train=np.array([0, 2, 3]), val=np.array([1]), test=np.array([4]))
    summary = split_summary(Y, split)
    assert summary["train"] == {"rows": 3, "fake": 1, "fake_share": pytest.approx(1 / 3)}
    assert summary["val"]["fake_share"] == 1.0 and summary["test"]["fake"] == 0
