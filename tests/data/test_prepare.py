"""Tests for cvx.data.prepare: split plans, model data and row selection."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cvx.config import SplitConfig, load_config
from cvx.data.prepare import make_model_data, make_split_plan, split_key, take, with_rows
from cvx.data.split import SplitIndices, check_no_group_overlap


@pytest.fixture
def config(tmp_path):
    """Grouped config sized for the 90-posting corpus."""
    path = tmp_path / "c.yaml"
    path.write_text(
        "experiment_id: t\nseed: 42\n"
        "paths: {data_csv: x.csv, runs_dir: runs, reports_dir: reports}\n"
        "preprocessing: p_cvx\n"
        "split: {method: grouped_stratified, ratios: [0.6, 0.2, 0.2], near_dup_threshold: 0.95, n_folds: 5}\n"  # noqa: E501
        "tfidf: {ngram_range: [1, 1], min_df: 1, max_df: 1.0, max_features: null, sublinear_tf: false}\n",  # noqa: E501
        encoding="utf-8",
    )
    return load_config(path)


def test_grouped_plan_removes_exact_duplicates_and_groups(corpus, config) -> None:
    doubled = pd.concat([corpus, corpus.assign(job_id=corpus["job_id"] + 1000)], ignore_index=True)
    plan = make_split_plan(doubled, config)
    assert len(plan.postings) == len(corpus)  # the copies differ only in job_id
    check_no_group_overlap(plan.split, plan.groups)


def test_random_plan_keeps_all_rows(corpus, config) -> None:
    random_config = replace(config, split=SplitConfig("random_stratified", [0.8, 0.1, 0.1]))
    plan = make_split_plan(corpus, random_config)
    assert len(plan.postings) == len(corpus) and plan.groups is None
    assert len(plan.split.test) == 9


def test_model_data(corpus, config) -> None:
    data = make_model_data(make_split_plan(corpus, config), config)
    assert list(data.X.columns) == ["text"]
    assert data.y.sum() == 30 and data.ids[0] == "1"
    assert not data.prevectorized


def test_split_key_ignores_preprocessing(config) -> None:
    assert split_key(config) == split_key(replace(config, preprocessing="p_cv"))


def test_take_and_with_rows() -> None:
    frame = pd.DataFrame({"text": ["a", "b", "c"]})
    assert take(frame, np.array([2, 0]))["text"].tolist() == ["c", "a"]
    assert take(sparse.identity(3, format="csr"), np.array([1])).toarray().tolist() == [[0, 1, 0]]
    from cvx.data.prepare import ModelData

    data = ModelData(
        frame,
        np.array([0, 1, 1]),
        np.array(["x", "y", "z"]),
        SplitIndices(np.array([0]), np.array([1]), np.array([2])),
        np.array([5, 6, 7]),
    )
    picked = with_rows(data, np.array([1, 1, 2]), data.split)
    assert picked.ids.tolist() == ["y", "y", "z"] and picked.groups is None
    assert picked.X.index.tolist() == [0, 1, 2]
