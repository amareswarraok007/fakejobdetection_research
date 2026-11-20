"""The five classifiers and their tuning grids (PRD Section 8.2, E3; speed mode §9.1).

| id         | model                                   | grid (max 3 values each)   |
| ---------- | --------------------------------------- | -------------------------- |
| nb         | Multinomial Naive Bayes                 | alpha in {0.1, 0.5, 1}     |
| linear_svm | Linear SVM + sigmoid calibration        | C in {0.1, 1, 10}          |
| rf         | Random Forest, 300 trees                | none (fixed settings)      |
| lr         | Logistic Regression, L2, balanced       | C in {0.1, 1, 10}          |
| xgb        | XGBoost, hist, lr 0.05, colsample 0.3,  | max_depth in {4, 6, 8}     |
|            | max_bin 64 (speed mode)                 |                            |

A config can override a grid with its ``grids`` key (E0 uses this).
"""

import os
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "nb": {"alpha": [0.1, 0.5, 1.0]},
    "linear_svm": {"C": [0.1, 1.0, 10.0]},
    "rf": {},
    "lr": {"C": [0.1, 1.0, 10.0]},
    "xgb": {"max_depth": [4, 6, 8]},
}
LR_MAX_ITER = 2000  # fixed; the paper tuned it but gives no value (owner-approved)
SVM_MAX_ITER = 5000  # higher than the default 1000 to avoid convergence warnings
CALIBRATION_FOLDS = 5
RF_TREES = 300
XGB_MAX_TREES = 600
XGB_LEARNING_RATE = 0.05
XGB_EARLY_STOPPING_ROUNDS = 50
# Speed mode (owner-approved 20 Nov 2026, PRD §9.1): with ~20,000 sparse word features,
# each tree samples 30% of the features and uses 64 histogram bins instead of 256.
XGB_COLSAMPLE_BYTREE = 0.3
XGB_MAX_BIN = 64


def n_jobs() -> int:
    """CPU threads for RF and XGBoost: env ``CVX_N_JOBS`` if set, else all cores (-1).

    The thread count does not change results; it only changes speed.
    """
    return int(os.environ.get("CVX_N_JOBS", "-1"))


def grid_for(model_id: str, overrides: dict[str, dict[str, list[Any]]]) -> dict[str, list[Any]]:
    """The grid for ``model_id``: the config's override if given, else the default."""
    return overrides.get(model_id, DEFAULT_GRIDS[model_id])


def candidates(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Every combination of grid values; ``[{}]`` for an empty grid (fixed settings)."""
    return list(ParameterGrid(grid)) if grid else [{}]


def scale_pos_weight(y_train: np.ndarray) -> float:
    """Genuine count / fake count in the training labels (XGBoost's class weighting)."""
    fake = int(np.sum(y_train == 1))
    return float(np.sum(y_train == 0) / fake) if fake else 1.0


def calibration_cv(y_train: np.ndarray, groups_train: np.ndarray | None, seed: int) -> Any:
    """Folds for the SVM's probability calibration.

    With groups (grouped experiments) the folds keep each near-duplicate group together,
    so calibration never sees a copy of a posting it is calibrated on. Without groups
    (E0-style random split) scikit-learn's default stratified folds are used.
    """
    if groups_train is None:
        return CALIBRATION_FOLDS
    splitter = StratifiedGroupKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y_train)), y_train, groups_train))


def make_classifier(
    model_id: str,
    params: dict[str, Any],
    y_train: np.ndarray,
    seed: int,
    groups_train: np.ndarray | None = None,
) -> Any:
    """Create one unfitted classifier.

    Inputs: model id, hyperparameters from the grid, the training labels (for XGBoost's
    ``scale_pos_weight``), seed, and training groups (for the SVM's grouped calibration).
    Output: an estimator with ``fit`` and ``predict_proba``.
    """
    if model_id == "nb":
        return MultinomialNB(**params)
    if model_id == "linear_svm":
        svm = LinearSVC(class_weight="balanced", max_iter=SVM_MAX_ITER, random_state=seed, **params)
        cv = calibration_cv(y_train, groups_train, seed)
        return CalibratedClassifierCV(svm, method="sigmoid", cv=cv)
    if model_id == "rf":
        return RandomForestClassifier(
            n_estimators=RF_TREES,
            class_weight="balanced_subsample",
            n_jobs=n_jobs(),
            random_state=seed,
            **params,
        )
    if model_id == "lr":
        # Default penalty is L2; default solver is lbfgs.
        return LogisticRegression(
            class_weight="balanced", max_iter=LR_MAX_ITER, random_state=seed, **params
        )
    if model_id == "xgb":
        return XGBClassifier(
            tree_method="hist",
            learning_rate=XGB_LEARNING_RATE,
            n_estimators=XGB_MAX_TREES,
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            max_bin=XGB_MAX_BIN,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight(y_train),
            random_state=seed,
            n_jobs=n_jobs(),
            **params,
        )
    raise ValueError(f"Unknown model id {model_id!r}")
