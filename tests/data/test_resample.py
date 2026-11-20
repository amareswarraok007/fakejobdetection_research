"""Tests for cvx.data.resample: E1a/E1b balance before the split; samplers."""

from dataclasses import replace

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler

from cvx.config import ResampleConfig, SplitConfig, TfidfConfig, load_config
from cvx.data.prepare import make_model_data, make_split_plan
from cvx.data.resample import apply_before_split, make_sampler


@pytest.fixture
def base(corpus, tmp_path):
    """E0-style config (random split) and its model data for the 90-posting corpus."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "experiment_id: t\nseed: 42\n"
        "paths: {data_csv: x.csv, runs_dir: runs, reports_dir: reports}\n"
        "preprocessing: p_cv\n"
        "split: {method: random_stratified, ratios: [0.8, 0.1, 0.1]}\n"
        "tfidf: {ngram_range: [1, 1], min_df: 1, max_df: 1.0, max_features: null, sublinear_tf: false}\n",  # noqa: E501
        encoding="utf-8",
    )
    config = load_config(path)
    return config, make_model_data(make_split_plan(corpus, config), config)


def test_make_sampler() -> None:
    assert isinstance(make_sampler("random_oversample", 42), RandomOverSampler)
    assert isinstance(make_sampler("smote", 42), SMOTE)
    with pytest.raises(ValueError):
        make_sampler("other", 42)


def test_no_resampling_returns_same_data(base) -> None:
    config, data = base
    assert apply_before_split(data, config) is data
    train_only = replace(config, resample=ResampleConfig("random_oversample", "train_only"))
    assert apply_before_split(data, train_only) is data  # handled inside the pipeline


def test_oversample_before_split_balances_everything(base) -> None:
    config, data = base
    e1a = apply_before_split(
        data, replace(config, resample=ResampleConfig("random_oversample", "before_split"))
    )
    assert len(e1a.y) == 120 and e1a.y.sum() == 60  # 60 genuine + 60 fake
    assert len(set(e1a.ids)) == 90  # fake postings are repeated, not invented
    assert e1a.y[e1a.split.test].mean() == pytest.approx(0.5, abs=0.1)


def test_smote_before_split_is_prevectorized(base) -> None:
    config, data = base
    e1b = apply_before_split(
        data, replace(config, resample=ResampleConfig("smote", "before_split"))
    )
    assert e1b.prevectorized and e1b.X.shape[0] == 120
    assert sum(i.startswith("smote_") for i in e1b.ids) == 30
    assert e1b.y.sum() == 60


def test_split_config_is_used(base) -> None:
    config, data = base
    other = replace(
        config,
        split=SplitConfig("random_stratified", [0.5, 0.25, 0.25]),
        resample=ResampleConfig("random_oversample", "before_split"),
        tfidf=TfidfConfig([1, 1], 1, 1.0, None, False),
    )
    result = apply_before_split(data, other)
    assert len(result.split.test) == 30
    assert np.array_equal(
        np.sort(np.concatenate([result.split.train, result.split.val, result.split.test])),
        np.arange(120),
    )
