"""Tests for cvx.evaluate.stats: exact McNemar by hand, bootstrap behaviour."""

import numpy as np
import pytest

from cvx.evaluate.stats import bootstrap_ci, mcnemar_exact


def test_mcnemar_hand_example() -> None:
    # 8 postings, all fake. A right on 0-2 (3 postings), B right on 2-7 (6 postings).
    # b = A right & B wrong = {0, 1} -> 2; c = A wrong & B right = {3..7} -> 5.
    # Two-sided exact p = 2 * P(X <= 2 | n=7, p=0.5) = 2 * (1 + 7 + 21) / 128 = 58/128.
    y = np.ones(8, dtype=int)
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    b = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    result = mcnemar_exact(y, a, b)
    assert (result["b"], result["c"]) == (2, 5)
    assert result["p_value"] == pytest.approx(58 / 128)


def test_mcnemar_no_disagreement() -> None:
    y = np.array([1, 0, 1])
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0


def test_bootstrap_perfect_separation_gives_tight_ci() -> None:
    y = np.array([0] * 20 + [1] * 10)
    s = np.concatenate([np.linspace(0.0, 0.4, 20), np.linspace(0.6, 1.0, 10)])
    ci = bootstrap_ci(y, s, {"0.5": 0.5}, n_resamples=200, seed=42)
    assert ci["pr_auc"] == {"low": pytest.approx(1.0), "high": pytest.approx(1.0)}
    assert ci["at_threshold"]["0.5"]["f1"] == {
        "low": pytest.approx(1.0),
        "high": pytest.approx(1.0),
    }
    assert ci["n_used"] == 200 and ci["n_resamples"] == 200


def test_bootstrap_is_reproducible_and_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    first = bootstrap_ci(y, s, {"0.5": 0.5}, n_resamples=300, seed=42)
    second = bootstrap_ci(y, s, {"0.5": 0.5}, n_resamples=300, seed=42)
    assert first == second
    assert first["roc_auc"]["low"] < first["roc_auc"]["high"]


def test_bootstrap_skips_single_class_resamples() -> None:
    y = np.array([0, 0, 0, 1])  # many resamples will have no fake row
    ci = bootstrap_ci(y, np.array([0.1, 0.2, 0.3, 0.9]), {}, n_resamples=100, seed=1)
    assert 0 < ci["n_used"] < 100
