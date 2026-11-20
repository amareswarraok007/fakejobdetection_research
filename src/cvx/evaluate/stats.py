"""Statistics on the test split (PRD Section 6.5): bootstrap CIs and McNemar's test.

- Bootstrap CI: draw the test set with replacement many times (1,000 in the configs),
  recompute each metric, and take the middle 95% of the values (percentile method).
  Thresholds stay fixed at the values chosen on validation.
- McNemar's test: do two models make different mistakes on the same test postings?
  Only the postings where exactly one model is right matter (b and c below).
  We use the exact binomial version, which is valid for any b + c.
- Wilcoxon signed-rank test (E6): are paired per-posting values (e.g. comp@5 of
  SHAP vs random) different, without assuming a normal distribution?
"""

from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from cvx.evaluate.metrics import confusion_counts, rates_from_counts, threshold_free

CI_RATE_METRICS = ("precision", "recall", "f1")


def _interval(values: list[float], level: float) -> dict[str, float | None]:
    """Percentile interval of ``values`` (None when there are no values)."""
    if not values:
        return {"low": None, "high": None}
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"low": float(low), "high": float(high)}


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, float],
    n_resamples: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap percentile CIs for PR-AUC, ROC-AUC and precision/recall/F1 at each threshold.

    A resample with only one class cannot give PR-AUC or ROC-AUC, so it is skipped;
    ``n_used`` reports how many resamples were kept.
    """
    y, scores = np.asarray(y).astype(int), np.asarray(scores)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {"pr_auc": [], "roc_auc": []}
    per_threshold = {name: {m: [] for m in CI_RATE_METRICS} for name in thresholds}
    n_used = 0
    for _ in range(n_resamples):
        rows = rng.integers(0, len(y), len(y))
        y_b, s_b = y[rows], scores[rows]
        if y_b.min() == y_b.max():
            continue
        n_used += 1
        for name, value in threshold_free(y_b, s_b).items():
            collected[name].append(value)
        for name, t in thresholds.items():
            rates = rates_from_counts(confusion_counts(y_b, s_b >= t))
            for metric in CI_RATE_METRICS:
                per_threshold[name][metric].append(rates[metric])
    return {
        "n_resamples": n_resamples,
        "n_used": n_used,
        "level": level,
        **{name: _interval(values, level) for name, values in collected.items()},
        "at_threshold": {
            name: {m: _interval(v, level) for m, v in metrics.items()}
            for name, metrics in per_threshold.items()
        },
    }


def mcnemar_exact(
    y: np.ndarray, predicted_a: np.ndarray, predicted_b: np.ndarray
) -> dict[str, Any]:
    """Exact McNemar test (two-sided binomial test on the discordant pairs).

    b = postings model A gets right and model B gets wrong; c = the reverse.
    Under "no difference", b ~ Binomial(b + c, 0.5). Output: b, c and the p-value.
    """
    y = np.asarray(y).astype(int)
    right_a = np.asarray(predicted_a).astype(int) == y
    right_b = np.asarray(predicted_b).astype(int) == y
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    p_value = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"test": "exact McNemar (binomial)", "b": b, "c": c, "p_value": p_value}


def wilcoxon_paired(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Two-sided paired Wilcoxon signed-rank test on ``first - second``.

    Pairs with a zero difference are dropped (SciPy's default). If every difference is
    zero the test cannot run; the p-value is then reported as 1.0.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    differences = first - second
    result: dict[str, Any] = {
        "test": "Wilcoxon signed-rank (two-sided)",
        "n": int(len(differences)),
        "median_difference": float(np.median(differences)) if len(differences) else None,
    }
    if not np.any(differences != 0):
        return {**result, "statistic": None, "p_value": 1.0}
    outcome = wilcoxon(first, second)
    return {**result, "statistic": float(outcome.statistic), "p_value": float(outcome.pvalue)}
