"""Pick the right SHAP method for a fitted pipeline (exact linear or Tree SHAP)."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from cvx.explain.core import Explanation
from cvx.explain.shap_linear import explain_linear, linear_shap
from cvx.explain.shap_tree import explain_tree, tree_shap


def model_kind(estimator: Any) -> str:
    """'linear' for Logistic Regression, 'tree' for XGBoost (others are not explained)."""
    classifier = estimator.named_steps["clf"]
    if isinstance(classifier, LogisticRegression):
        return "linear"
    if isinstance(classifier, XGBClassifier):
        return "tree"
    raise ValueError(f"No SHAP method for {type(classifier).__name__}")


def explain_shap(estimator: Any, X: Any, means: np.ndarray | None = None) -> list[Explanation]:
    """SHAP explanation per row; Logistic Regression needs the training ``means``."""
    if model_kind(estimator) == "linear":
        if means is None:
            raise ValueError("Linear SHAP needs the training feature means.")
        return explain_linear(estimator, X, means)
    return explain_tree(estimator, X)


def shap_matrix(estimator: Any, X: Any, means: np.ndarray | None = None) -> np.ndarray:
    """Raw SHAP values (rows x features), e.g. for global rankings."""
    if model_kind(estimator) == "linear":
        return linear_shap(estimator, X, means)[0]
    return tree_shap(estimator, X)[0]
