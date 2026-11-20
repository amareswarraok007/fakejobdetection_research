"""Tests for cvx.models.train: thresholds by hand, best-candidate choice, tuning paths."""

from dataclasses import replace

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from cvx.config import ResampleConfig, TfidfConfig, TuningConfig, load_config
from cvx.data.prepare import make_model_data, make_split_plan
from cvx.models.train import (
    assemble,
    choose_thresholds,
    cv_folds,
    make_head_factory,
    predict_scores,
    resolve_thresholds,
    select_best,
    train_model,
    train_val_fold,
)
from cvx.models.zoo import candidates


def test_choose_thresholds_hand_example() -> None:
    # Thresholds 0.1, 0.2, 0.4, 0.6, 0.9 give F1 = .571, .667, .8, .5, .667 -> t_f1 = 0.4.
    # Recall is 1 for thresholds <= 0.4 and 0.5 above -> highest with recall >= 0.9 is 0.4.
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    assert choose_thresholds(y, s) == {"t_f1": 0.4, "t_r90": 0.4}


def test_t_r90_is_the_highest_threshold_with_recall_090() -> None:
    # 10 fakes: 9 of them score >= 0.55, the 10th scores 0.3. Recall >= 0.9 needs 9 caught,
    # so the HIGHEST such threshold is 0.55 (not the lowest score, which gives recall 1).
    fake = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.3]
    genuine = [0.88, 0.5, 0.2, 0.1]
    y = np.array([1] * 10 + [0] * 4)
    assert choose_thresholds(y, np.array(fake + genuine))["t_r90"] == 0.55


def test_select_best_uses_mean_over_folds() -> None:
    rows = [
        {"params": {"C": 1}, "fold": 0, "score": 0.9},
        {"params": {"C": 1}, "fold": 1, "score": 0.5},
        {"params": {"C": 10}, "fold": 0, "score": 0.8},
        {"params": {"C": 10}, "fold": 1, "score": 0.8},
    ]
    assert select_best(rows) == ({"C": 10}, pytest.approx(0.8))


def test_resolve_thresholds() -> None:
    chosen = {"t_f1": 0.42, "t_r90": 0.17}
    assert resolve_thresholds([0.5, "t_f1", "t_r90"], chosen) == {
        "0.5": 0.5,
        "t_f1": 0.42,
        "t_r90": 0.17,
    }


@pytest.fixture
def grouped_config(tmp_path):
    """A grouped-split config sized for the 90-posting corpus."""
    text = """
experiment_id: t
seed: 42
paths: {data_csv: x.csv, runs_dir: runs, reports_dir: reports}
preprocessing: p_cvx
split: {method: grouped_stratified, ratios: [0.6, 0.2, 0.2], near_dup_threshold: 0.95, n_folds: 5}
tfidf: {ngram_range: [1, 2], min_df: 1, max_df: 1.0, max_features: 500, sublinear_tf: true}
models: [lr]
tuning: {method: grouped_cv, cv_folds: 3, scoring: average_precision}
"""
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


@pytest.fixture
def grouped_data(corpus, grouped_config):
    """Model data for the corpus under the grouped config."""
    return make_model_data(make_split_plan(corpus, grouped_config), grouped_config)


def test_cv_folds_fit_features_on_fold_rows_only(grouped_data, grouped_config) -> None:
    folds = cv_folds(grouped_data, grouped_config)
    assert len(folds) == 3
    n_train = len(grouped_data.split.train)
    for fold in folds:
        assert fold.X_fit.shape[0] + fold.X_eval.shape[0] == n_train
        # The vocabulary comes from the fold's own training rows, so folds differ.
        assert fold.features is not folds[0].features or fold is folds[0]


@pytest.mark.parametrize("model_id", ["lr", "xgb", "linear_svm"])
def test_train_model_grouped_cv(grouped_data, grouped_config, model_id: str) -> None:
    grid = {"lr": {"C": [0.1, 10.0]}, "xgb": {"max_depth": [2, 3]}, "linear_svm": {"C": [1.0]}}
    cands = candidates(grid[model_id])
    trained = train_model(
        model_id,
        make_head_factory(model_id, grouped_config),
        cands,
        cv_folds(grouped_data, grouped_config),
        train_val_fold(grouped_data, grouped_config),
        grouped_config.tuning,
    )
    assert len(trained.cv_rows) == (3 * len(cands) if len(cands) > 1 else 0)
    assert len(trained.val_scores) == len(grouped_data.split.val)
    assert set(trained.thresholds) == {"t_f1", "t_r90"}
    # The final estimator predicts straight from the raw model input table.
    scores = predict_scores(trained.estimator, grouped_data.X.iloc[grouped_data.split.test])
    assert len(scores) == len(grouped_data.split.test)


def test_assembled_model_equals_one_pipeline(grouped_data, grouped_config) -> None:
    # Fitting features once and reusing them must give the same model as a plain pipeline.
    fold = train_val_fold(grouped_data, grouped_config)
    head = make_head_factory("lr", grouped_config)({"C": 1.0}, fold.y_fit, None).fit(
        fold.X_fit, fold.y_fit
    )
    shared = assemble(fold.features, head)
    from sklearn.base import clone

    plain = Pipeline([("features", clone(fold.features)), ("clf", clone(head))])
    train = grouped_data.split.train
    plain.fit(grouped_data.X.iloc[train], grouped_data.y[train])
    test_rows = grouped_data.X.iloc[grouped_data.split.test]
    assert np.allclose(predict_scores(shared, test_rows), predict_scores(plain, test_rows))


def test_train_model_validation_reuses_best_fit(grouped_data, grouped_config) -> None:
    tuning = TuningConfig(method="validation", scoring="roc_auc")
    config = replace(grouped_config, tuning=tuning)
    trained = train_model(
        "lr",
        make_head_factory("lr", config),
        candidates({"C": [0.1, 1.0]}),
        None,
        train_val_fold(grouped_data, config),
        tuning,
    )
    assert [r["fold"] for r in trained.cv_rows] == ["val", "val"]
    assert trained.best_params in ({"C": 0.1}, {"C": 1.0})


def test_train_only_resampling_is_inside_the_model(grouped_data, grouped_config) -> None:
    config = replace(
        grouped_config,
        resample=ResampleConfig(method="smote", when="train_only"),
        tfidf=TfidfConfig([1, 1], 1, 1.0, None, False),
    )
    fold = train_val_fold(grouped_data, config)
    head = make_head_factory("lr", config)({"C": 1.0}, fold.y_fit, None)
    assert isinstance(head, ImbPipeline)
    estimator = assemble(fold.features, head.fit(fold.X_fit, fold.y_fit))
    assert [name for name, _ in estimator.steps] == ["features", "resample", "clf"]
    # The sampler acts only during fit: predicting gives one score per input row.
    test_rows = grouped_data.X.iloc[grouped_data.split.test]
    assert len(predict_scores(estimator, test_rows)) == len(test_rows)
