"""Exact SHAP for Logistic Regression (PRD E5; CLAUDE.md §6).

For a linear model on the log-odds scale, the SHAP value of feature i is
    phi_i = w_i * (x_i - mean of x_i in the training data)
and the base value is  intercept + sum_i w_i * mean_i.  So the base value plus all
phi_i equals the model's log-odds for that posting. ``check_against_shap`` confirms
the formula matches ``shap.LinearExplainer`` (the PRD requires a match within 1e-6).
"""

from typing import Any

import numpy as np
import shap
from scipy import sparse

from cvx.explain.core import Explanation, explanation_from_row, feature_names, transform_features


def training_means(estimator: Any, X_train: Any) -> np.ndarray:
    """Mean of every model feature over the training rows (input to the formula)."""
    return np.asarray(transform_features(estimator, X_train).mean(axis=0)).ravel()


def linear_shap(
    estimator: Any, X: Any, means: np.ndarray
) -> tuple[np.ndarray, float, sparse.csr_matrix]:
    """SHAP values (rows x features), base value and the feature matrix, by the formula."""
    classifier = estimator.named_steps["clf"]
    weights = np.asarray(classifier.coef_).ravel()
    matrix = transform_features(estimator, X)
    phi = (matrix.toarray() - means) * weights
    base = float(np.ravel(classifier.intercept_)[0] + weights @ means)
    return phi, base, matrix


def explain_linear(estimator: Any, X: Any, means: np.ndarray) -> list[Explanation]:
    """One ``Explanation`` per row of ``X`` for a Logistic Regression pipeline."""
    phi, base, matrix = linear_shap(estimator, X, means)
    scores = np.asarray(estimator.predict_proba(X))[:, 1]
    names = feature_names(estimator)
    return [
        explanation_from_row("shap", names, phi[i], matrix[i], float(scores[i]), base)
        for i in range(matrix.shape[0])
    ]


def check_against_shap(estimator: Any, X: Any, means: np.ndarray) -> float:
    """Largest absolute difference between our formula and ``shap.LinearExplainer``.

    The training mean is given to SHAP as its one-row background, so both use the
    same reference point (SHAP's default would subsample 100 training rows instead).
    """
    phi, base, matrix = linear_shap(estimator, X, means)
    explainer = shap.LinearExplainer(estimator.named_steps["clf"], means.reshape(1, -1))
    reference = explainer.shap_values(matrix)
    reference = reference.toarray() if sparse.issparse(reference) else np.asarray(reference)
    base_gap = abs(float(np.ravel(explainer.expected_value)[0]) - base)
    return float(max(np.abs(reference - phi).max(), base_gap))
