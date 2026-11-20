"""Error analysis (PRD E7): the most confident mistakes with their top-5 SHAP features.

False negatives = fake postings the model missed (lowest scores first);
false positives = genuine postings flagged as fake (highest scores first).
The ``cause`` column is left empty for the owner to group the errors by hand.
"""

from typing import Any

import numpy as np

from cvx.explain.context import XaiContext
from cvx.explain.dispatch import explain_shap

ERROR_COLUMNS = [
    "model",
    "error_type",
    "row_id",
    "label",
    "score",
    "threshold",
    "top5_shap",
    "cause",
]


def confident_errors(
    y: np.ndarray, scores: np.ndarray, threshold: float, n: int
) -> dict[str, np.ndarray]:
    """Indices of the n most confident false negatives and false positives."""
    predicted = scores >= threshold
    fn = np.flatnonzero((y == 1) & ~predicted)
    fp = np.flatnonzero((y == 0) & predicted)
    return {
        "false_negative": fn[np.argsort(scores[fn], kind="stable")][:n],
        "false_positive": fp[np.argsort(-scores[fp], kind="stable")][:n],
    }


def _top5(features: dict[str, float]) -> str:
    """'feature (+0.123); …' for the five largest absolute contributions."""
    items = sorted(features.items(), key=lambda item: (-abs(item[1]), item[0]))[:5]
    return "; ".join(f"{name} ({value:+.3f})" for name, value in items)


def error_rows(ctx: XaiContext, model_id: str, n: int) -> list[dict[str, Any]]:
    """Rows for errors.csv for one model at its ``t_f1`` threshold."""
    loaded = ctx.store.models[model_id]
    test = ctx.test_rows()
    scores = ctx.scores(model_id, test)
    threshold = loaded.info["thresholds"]["t_f1"]
    rows = []
    for error_type, picked in confident_errors(ctx.data.y[test], scores, threshold, n).items():
        if len(picked) == 0:
            continue
        explanations = explain_shap(loaded.estimator, ctx.rows(test[picked]), loaded.means)
        for index, explanation in zip(picked, explanations, strict=True):
            rows.append(
                {
                    "model": model_id,
                    "error_type": error_type,
                    "row_id": str(ctx.data.ids[test[index]]),
                    "label": int(ctx.data.y[test[index]]),
                    "score": float(scores[index]),
                    "threshold": threshold,
                    "top5_shap": _top5(explanation.features),
                    "cause": "",
                }
            )
    return rows
