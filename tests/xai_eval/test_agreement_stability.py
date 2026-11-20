"""Tests for cvx.xai_eval.agreement and cvx.xai_eval.stability on hand examples."""

import pytest

from cvx.explain.core import Explanation
from cvx.xai_eval.agreement import agreement, jaccard
from cvx.xai_eval.stability import lime_stability, mean_pairwise_jaccard


def test_jaccard() -> None:
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0


def test_agreement_hand_example() -> None:
    shap = Explanation("shap", 0.9, 0.0, words={"fee": 0.9, "pay": 0.5, "team": -0.4})
    lime = Explanation("lime", 0.9, None, words={"fee": 0.3, "pay": -0.2, "now": 0.1})
    result = agreement(shap, lime)
    # top-5 sets: {fee, pay, team} vs {fee, pay, now} -> 2 shared of 4.
    assert result["jaccard_5"] == pytest.approx(2 / 4) and result["jaccard_10"] == pytest.approx(
        0.5
    )
    # Shared words fee (+,+) and pay (+,-): 1 of 2 push the same way.
    assert result["same_direction"] == pytest.approx(0.5)
    assert -1 <= result["spearman"] <= 1


def test_stability() -> None:
    assert mean_pairwise_jaccard([["a", "b"], ["a", "b"], ["a", "c"]]) == pytest.approx(
        (1 + 1 / 3 + 1 / 3) / 3
    )
    same = [Explanation("lime", 0.5, None, words={"x": 1.0, "y": 0.5}) for _ in range(3)]
    assert lime_stability(same) == 1.0
