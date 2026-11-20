"""Tests for cvx.data.dedup on hand-made vectors and postings with known answers."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cvx.data.dedup import (
    drop_exact_duplicates,
    exact_duplicate_mask,
    group_summary,
    groups_from_pairs,
    near_duplicate_groups,
    similar_pairs,
)


def test_exact_duplicates_ignore_job_id(tiny_postings: pd.DataFrame) -> None:
    # Row 1 copies row 0 (including a NaN benefits field) with a different job_id.
    # Row 3 has the same text as row 2 but a different location, so it is NOT exact.
    assert exact_duplicate_mask(tiny_postings).tolist() == [False, True, False, False, False, False]


def test_drop_exact_duplicates_keeps_first(tiny_postings: pd.DataFrame) -> None:
    kept, n_removed = drop_exact_duplicates(tiny_postings)
    assert n_removed == 1
    assert list(kept.index) == [0, 2, 3, 4, 5]


# Unit vectors with known cosines: c(0,1)=1, c(0,3)=c(1,3)=0.6, c(2,3)=0.8, c(0,2)=0.
VECTORS = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.95, [[0, 1]]),
        (0.7, [[0, 1], [2, 3]]),
        (0.5, [[0, 1], [0, 3], [1, 3], [2, 3]]),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_similar_pairs(threshold: float, expected: list[list[int]], batch_size: int) -> None:
    pairs = similar_pairs(VECTORS, threshold, batch_size=batch_size)
    assert sorted(map(tuple, pairs.tolist())) == sorted(map(tuple, expected))


def test_groups_chain_transitively() -> None:
    # 0~1 and 1~2 link 0, 1, 2 into one group even though 0 and 2 are not linked directly.
    groups = groups_from_pairs(5, np.array([[0, 1], [1, 2]]))
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3


def test_groups_with_no_pairs() -> None:
    assert len(set(groups_from_pairs(3, np.empty((0, 2))).tolist())) == 3


def test_near_duplicate_groups_on_text() -> None:
    texts = ["sales manager wanted", "sales manager wanted", "python developer needed", "chef"]
    groups = near_duplicate_groups(texts, threshold=0.95)
    assert groups[0] == groups[1]
    assert len({groups[0], groups[2], groups[3]}) == 3


def test_near_duplicate_groups_empty() -> None:
    assert near_duplicate_groups([], threshold=0.95).size == 0


def test_group_summary() -> None:
    groups = np.array([0, 0, 1, 2, 2, 2])
    labels = np.array([0, 0, 1, 0, 1, 1])
    assert group_summary(groups, labels) == {
        "rows": 6,
        "groups": 3,
        "groups_with_more_than_one_row": 2,
        "rows_in_such_groups": 5,
        "largest_group_size": 3,
        "groups_with_mixed_labels": 1,
    }
