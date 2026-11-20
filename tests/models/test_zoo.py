"""Tests for cvx.models.zoo: grids, weights, grouped calibration, and that each model fits."""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from cvx.config import MAX_GRID_VALUES, MODEL_IDS
from cvx.models.zoo import (
    DEFAULT_GRIDS,
    calibration_cv,
    candidates,
    grid_for,
    make_classifier,
    scale_pos_weight,
)


def test_default_grids_respect_speed_mode() -> None:
    assert set(DEFAULT_GRIDS) == set(MODEL_IDS)
    for grid in DEFAULT_GRIDS.values():
        assert all(len(values) <= MAX_GRID_VALUES for values in grid.values())
    assert DEFAULT_GRIDS["lr"] == {"C": [0.1, 1.0, 10.0]}


def test_grid_override_and_candidates() -> None:
    assert grid_for("lr", {"lr": {"C": [1.0]}}) == {"C": [1.0]}
    assert grid_for("nb", {}) == DEFAULT_GRIDS["nb"]
    assert candidates({}) == [{}]
    assert len(candidates({"a": [1, 2], "b": [3, 4, 5]})) == 6


def test_scale_pos_weight() -> None:
    assert scale_pos_weight(np.array([0, 0, 0, 1])) == 3.0
    assert scale_pos_weight(np.array([0, 0])) == 1.0


def test_calibration_folds_keep_groups_together() -> None:
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(10), 4)
    folds = calibration_cv(y, groups, seed=42)
    assert len(folds) == 5
    for fit_part, eval_part in folds:
        assert not set(groups[fit_part]) & set(groups[eval_part])
    assert calibration_cv(y, None, seed=42) == 5  # E0-style: default stratified folds


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_every_model_fits_and_gives_probabilities(model_id: str) -> None:
    rng = np.random.default_rng(42)
    X = rng.random((60, 5))
    y = (X[:, 0] > 0.5).astype(int)
    groups = np.arange(60) // 2
    params = candidates(DEFAULT_GRIDS[model_id])[0]
    model = make_classifier(model_id, params, y, seed=42, groups_train=groups)
    fit_args = {"eval_set": [(X, y)], "verbose": False} if model_id == "xgb" else {}
    model.fit(X, y, **fit_args)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (60,) and ((proba >= 0) & (proba <= 1)).all()
    if model_id == "linear_svm":
        assert isinstance(model, CalibratedClassifierCV) and isinstance(model.cv, list)


def test_xgboost_speed_settings() -> None:
    model = make_classifier("xgb", {"max_depth": 4}, np.array([0, 1]), seed=42)
    params = model.get_params()
    assert params["colsample_bytree"] == 0.3 and params["max_bin"] == 64
    assert params["learning_rate"] == 0.05 and params["n_estimators"] == 600
