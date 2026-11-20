"""Tree SHAP for XGBoost (PRD E5; CLAUDE.md §6).

``shap.TreeExplainer`` computes exact SHAP values for tree models on the margin
(log-odds) scale. Check: base value + sum of SHAP values = the model's margin,
within 1e-4 (``additivity_error``). Sparse TF-IDF zeros count as "missing" in
XGBoost; TreeExplainer uses XGBoost's own routine, so it treats them the same way.
"""

from typing import Any

import numpy as np
import shap
from scipy import sparse

from cvx.explain.core import Explanation, explanation_from_row, feature_names, transform_features


def tree_shap(estimator: Any, X: Any) -> tuple[np.ndarray, float, sparse.csr_matrix]:
    """SHAP values (rows x features), base value and the feature matrix."""
    matrix = transform_features(estimator, X)
    explainer = shap.TreeExplainer(estimator.named_steps["clf"])
    phi = np.asarray(explainer.shap_values(matrix))
    return phi, float(np.ravel(explainer.expected_value)[0]), matrix


def model_margin(estimator: Any, X: Any) -> np.ndarray:
    """XGBoost raw output (log-odds) for each row."""
    matrix = transform_features(estimator, X)
    return np.asarray(estimator.named_steps["clf"].predict(matrix, output_margin=True))


def additivity_error(estimator: Any, X: Any) -> float:
    """Largest |base + sum(SHAP) - margin| over the rows (should be < 1e-4)."""
    phi, base, _ = tree_shap(estimator, X)
    return float(np.abs(base + phi.sum(axis=1) - model_margin(estimator, X)).max())


def explain_tree(estimator: Any, X: Any) -> list[Explanation]:
    """One ``Explanation`` per row of ``X`` for an XGBoost pipeline."""
    phi, base, matrix = tree_shap(estimator, X)
    scores = np.asarray(estimator.predict_proba(X))[:, 1]
    names = feature_names(estimator)
    return [
        explanation_from_row("shap", names, phi[i], matrix[i], float(scores[i]), base)
        for i in range(matrix.shape[0])
    ]
