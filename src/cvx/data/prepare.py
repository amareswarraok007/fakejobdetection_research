"""Turn raw postings into what training needs: model input, labels, ids, split, groups.

Two steps, so work can be shared between the variants of one experiment:
1. ``make_split_plan``: which rows are used and how they are split (with duplicate
   removal and near-duplicate groups for the grouped split; all rows for the random one).
2. ``make_model_data``: the model input table for one preprocessing / feature setting.
"""

import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from cvx.config import Config
from cvx.data import clean, dedup, split
from cvx.data.load import ID_COLUMN, LABEL_COLUMN
from cvx.features.build import make_model_input

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitPlan:
    """Rows used (index reset to 0..n-1), their split, and near-duplicate groups (or None)."""

    postings: pd.DataFrame
    split: split.SplitIndices
    groups: np.ndarray | None


@dataclass(frozen=True)
class ModelData:
    """Everything a model needs. ``X`` is a DataFrame, or a sparse matrix if prevectorized."""

    X: Any
    y: np.ndarray
    ids: np.ndarray
    split: split.SplitIndices
    groups: np.ndarray | None
    prevectorized: bool = False


def take(X: Any, positions: np.ndarray) -> Any:
    """Rows of ``X`` at ``positions`` (works for DataFrames and sparse matrices)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[positions]
    return X[positions]


def split_key(config: Config) -> tuple[Any, ...]:
    """Settings that define a split plan; variants with the same key share one plan."""
    s = config.split
    return (s.method, tuple(s.ratios), s.near_dup_threshold, s.n_folds, config.seed)


def make_split_plan(df: pd.DataFrame, config: Config) -> SplitPlan:
    """Build the split described by ``config.split``.

    Random split (P-CV reproduction): all rows, stratified random split, no groups.
    Grouped split: exact duplicates removed, near-duplicates grouped on the P-CVX text
    without labels, then grouped stratified split; a group never crosses splits.
    """
    s = config.split
    if s.method == "random_stratified":
        postings = df.reset_index(drop=True)
        y = postings[LABEL_COLUMN].to_numpy()
        return SplitPlan(
            postings, split.random_stratified_split(y, tuple(s.ratios), config.seed), None
        )
    postings, n_removed = dedup.drop_exact_duplicates(df)
    postings = postings.reset_index(drop=True)
    logger.info("Removed %d exact duplicates", n_removed)
    texts = clean.p_cvx(postings, with_labels=False).tolist()
    groups = dedup.near_duplicate_groups(texts, s.near_dup_threshold)
    y = postings[LABEL_COLUMN].to_numpy()
    indices = split.grouped_stratified_split(y, groups, tuple(s.ratios), s.n_folds, config.seed)
    return SplitPlan(postings, indices, groups)


def make_model_data(plan: SplitPlan, config: Config) -> ModelData:
    """Model input for ``plan.postings`` with the config's preprocessing and features."""
    X = make_model_input(
        plan.postings,
        config.preprocessing,
        config.features.metadata,
        tuple(config.features.extra_categories),
    )
    return ModelData(
        X=X.reset_index(drop=True),
        y=plan.postings[LABEL_COLUMN].to_numpy().astype(int),
        ids=plan.postings[ID_COLUMN].astype(str).to_numpy(),
        split=plan.split,
        groups=plan.groups,
    )


def with_rows(data: ModelData, positions: np.ndarray, new_split: split.SplitIndices) -> ModelData:
    """A copy of ``data`` keeping only ``positions`` (repeats allowed) with a new split."""
    X = take(data.X, positions)
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    return replace(
        data, X=X, y=data.y[positions], ids=data.ids[positions], split=new_split, groups=None
    )


def is_sparse(X: Any) -> bool:
    """True when ``X`` is a scipy sparse matrix."""
    return sparse.issparse(X)
