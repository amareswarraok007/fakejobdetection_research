"""Tests for cvx.explain.core: per-word scores, top words, global ranking (hand examples)."""

import numpy as np
import pytest
from scipy import sparse

from cvx.explain.core import (
    Explanation,
    explanation_from_row,
    global_ranking,
    top_words,
    words_from_ngrams,
)


def test_phrase_score_is_split_equally() -> None:
    # 'registration fee' = 0.6 -> 0.3 each; 'fee' alone adds 0.1 -> fee = 0.4.
    words = words_from_ngrams({"registration fee": 0.6, "fee": 0.1, "team": -0.2})
    assert words == {"registration": pytest.approx(0.3), "fee": pytest.approx(0.4), "team": -0.2}


def test_explanation_from_row_hand_example() -> None:
    names = np.array(["text__fee", "text__no logo", "text__team", "numeric__has_company_logo"])
    phi = np.array([0.5, 0.2, -0.3, -0.1])
    # The posting contains 'fee' and 'no logo' but not 'team'.
    x_row = sparse.csr_matrix(np.array([[0.7, 0.4, 0.0, 0.0]]))
    exp = explanation_from_row("shap", names, phi, x_row, score=0.8, base_value=-1.0)
    assert exp.features == {"fee": 0.5, "no logo": 0.2, "numeric__has_company_logo": -0.1}
    assert exp.words == {"fee": 0.5, "no": pytest.approx(0.1), "logo": pytest.approx(0.1)}
    assert exp.absent_total == pytest.approx(-0.3)  # 'team' is absent


def test_top_words_by_direction() -> None:
    exp = Explanation("shap", 0.5, None, words={"a": 0.5, "b": -0.9, "c": 0.1, "d": -0.2})
    assert top_words(exp, 2, toward=1) == ["a", "c"]
    assert top_words(exp, 2, toward=0) == ["b", "d"]
    assert top_words(exp, 2) == ["b", "a"]


def test_global_ranking() -> None:
    names = np.array(["x", "y", "z"])
    phi = np.array([[0.1, -2.0, 0.0], [0.3, 1.0, 0.0]])  # mean |phi| = 0.2, 1.5, 0
    ranking = global_ranking(names, phi, top=2)
    assert [r["feature"] for r in ranking] == ["y", "x"]
    assert ranking[0]["mean_abs_shap"] == pytest.approx(1.5)
