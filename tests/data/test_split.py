"""Tests for cvx.data.split. The key test: no group ever appears in two splits."""

import numpy as np
import pytest

from cvx.data.split import (
    SplitIndices,
    check_no_group_overlap,
    grouped_stratified_split,
    random_stratified_split,
)


def _covers_all_once(split: SplitIndices, n_rows: int) -> bool:
    """Every row position is in exactly one split."""
    joined = np.concatenate([split.train, split.val, split.test])
    return sorted(joined.tolist()) == list(range(n_rows))


# ---------------------------------------------------------------- random (P-CV)


def test_random_split_sizes_and_stratification() -> None:
    y = np.array([0] * 90 + [1] * 10)
    split = random_stratified_split(y, (0.8, 0.1, 0.1), seed=42)
    assert (len(split.train), len(split.val), len(split.test)) == (80, 10, 10)
    assert (y[split.train].sum(), y[split.val].sum(), y[split.test].sum()) == (8, 1, 1)
    assert _covers_all_once(split, 100)


def test_random_split_is_reproducible() -> None:
    y = np.array([0] * 90 + [1] * 10)
    a = random_stratified_split(y, seed=42)
    b = random_stratified_split(y, seed=42)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.val, b.val)


# ---------------------------------------------------------------- grouped (P-CVX)


def _grouped_data() -> tuple[np.ndarray, np.ndarray]:
    """60 rows in 30 groups of two; groups 0-5 are fake (12 fake rows).

    Group 29 is mixed (one fake, one genuine) to check mixed groups also stay together.
    """
    groups = np.repeat(np.arange(30), 2)
    y = (groups < 6).astype(int)
    y[59] = 1
    return y, groups


def test_no_group_crosses_splits() -> None:
    y, groups = _grouped_data()
    split = grouped_stratified_split(y, groups, ratios=(0.6, 0.2, 0.2), n_folds=5, seed=42)
    train, val, test = (set(groups[i].tolist()) for i in (split.train, split.val, split.test))
    assert not (train & val) and not (train & test) and not (val & test)
    assert _covers_all_once(split, len(y))


def test_grouped_split_keeps_fakes_in_every_split() -> None:
    y, groups = _grouped_data()
    split = grouped_stratified_split(y, groups, ratios=(0.6, 0.2, 0.2), n_folds=5, seed=42)
    assert all(y[idx].sum() >= 1 for idx in (split.train, split.val, split.test))


def test_grouped_split_fold_counts() -> None:
    # 30 groups of 2 in 5 folds -> 6 groups (12 rows) per fold: 3 train folds, 1 val, 1 test.
    y, groups = _grouped_data()
    split = grouped_stratified_split(y, groups, ratios=(0.6, 0.2, 0.2), n_folds=5, seed=42)
    assert (len(split.train), len(split.val), len(split.test)) == (36, 12, 12)


def test_grouped_split_is_reproducible() -> None:
    y, groups = _grouped_data()
    a = grouped_stratified_split(y, groups, ratios=(0.6, 0.2, 0.2), n_folds=5, seed=42)
    b = grouped_stratified_split(y, groups, ratios=(0.6, 0.2, 0.2), n_folds=5, seed=42)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.val, b.val)


def test_ratios_must_fit_whole_folds() -> None:
    y, groups = _grouped_data()
    with pytest.raises(ValueError, match="whole numbers of folds"):
        grouped_stratified_split(y, groups, ratios=(0.7, 0.15, 0.15), n_folds=5)


def test_check_no_group_overlap_detects_leak() -> None:
    leaky = SplitIndices(train=np.array([0]), val=np.array([1]), test=np.array([2]))
    with pytest.raises(AssertionError, match="more than one split"):
        check_no_group_overlap(leaky, np.array([7, 7, 8]))


def test_as_labels() -> None:
    split = SplitIndices(train=np.array([0, 2]), val=np.array([1]), test=np.array([3]))
    assert split.as_labels(4).tolist() == ["train", "val", "train", "test"]
