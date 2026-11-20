"""PRD M5 check: comp@k, suff@k and AOPC on a toy model where the right answer is known.

Toy model: p(fake) = 0.1 + 0.4 * [contains 'fee'] + 0.4 * [contains 'urgent'].
For the posting 'pay fee urgent now team' the model predicts fake with p = 0.9.
A perfect explanation ranks ['fee', 'urgent', ...] first:
  remove top-1 ('fee')          -> p = 0.5 -> comp@1 = 0.9 - 0.5 = 0.4
  remove top-3 (fee, urgent, x) -> p = 0.1 -> comp@3 = 0.8
  keep only top-1 ('fee')       -> p = 0.5 -> suff@1 = 0.4
  keep only top-3               -> p = 0.9 -> suff@3 = 0.0
A wrong explanation (['team', 'now', 'pay']) gets comp@1 = comp@3 = 0 and suff = 0.8.
"""

import numpy as np
import pytest

from cvx.xai_eval.faithfulness import (
    eligible_words,
    evaluate_posting,
    keep_only,
    label_probability,
    random_rankings,
    remove_words,
)

TEXT = "title: pay fee urgent | description: now team fee"


def toy_predict(texts: list[str]) -> np.ndarray:
    """p(fake) from the presence of 'fee' and 'urgent'."""

    def one(text: str) -> float:
        words = set(text.replace(":", " ").split())
        return 0.1 + 0.4 * ("fee" in words) + 0.4 * ("urgent" in words)

    return np.array([one(t) for t in texts])


def test_remove_and_keep() -> None:
    assert remove_words(TEXT, ["fee"]) == "title: pay urgent | description: now team"
    assert remove_words(TEXT, ["title"]) == TEXT  # field labels are not words to remove
    assert keep_only(TEXT, ["fee", "now"]) == "fee now fee"
    assert eligible_words("title: a fee | description: fee now") == ["fee", "now"]


def test_known_answer() -> None:
    good = ["fee", "urgent", "pay", "now", "team"]
    bad = ["team", "now", "pay", "fee", "urgent"]
    result = evaluate_posting(toy_predict, TEXT, 1, {"good": good, "bad": bad}, [], ks=(1, 3))
    assert result["p_label"] == pytest.approx(0.9)
    assert result["good"]["comp"] == {1: pytest.approx(0.4), 3: pytest.approx(0.8)}
    assert result["good"]["suff"] == {1: pytest.approx(0.4), 3: pytest.approx(0.0)}
    assert result["good"]["aopc"] == pytest.approx(0.6)
    assert result["bad"]["comp"] == {1: pytest.approx(0.0), 3: pytest.approx(0.0)}
    assert result["bad"]["suff"][3] == pytest.approx(0.8)


def test_predicted_genuine_uses_genuine_probability() -> None:
    assert label_probability(np.array([0.2]), 0)[0] == pytest.approx(0.8)
    # For a genuine prediction, removing 'fee' RAISES p(genuine): comp is negative.
    result = evaluate_posting(toy_predict, TEXT, 0, {"m": ["fee"]}, [], ks=(1,))
    assert result["p_label"] == pytest.approx(0.1)
    assert result["m"]["comp"][1] == pytest.approx(0.1 - 0.5)


def test_random_baseline_is_averaged_and_seeded() -> None:
    words = eligible_words(TEXT)
    draws = random_rankings(words, draws=5, seed=42)
    assert draws == random_rankings(words, draws=5, seed=42) and len(draws) == 5
    result = evaluate_posting(toy_predict, TEXT, 1, {"good": words}, draws, ks=(1, 3))
    assert 0.0 <= result["random"]["comp"][1] <= 0.4
    assert set(result["random"]) == {"comp", "suff", "aopc"}
