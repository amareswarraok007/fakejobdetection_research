"""Find exact duplicates and group near-duplicate postings. PRD Section 6.4.

Why: if the same (or almost the same) posting sits in both the training and the
test set, the model is tested on something it has already seen, which inflates
scores. Exact copies are removed; near-copies are put in one *group* so the split
can keep each group inside a single split.

- Exact duplicate: identical in every column except ``job_id``; the first copy is kept.
- Near-duplicate: TF-IDF cosine similarity >= threshold (0.95 in base.yaml), measured
  on the P-CVX text without field labels, using plain word TF-IDF (scikit-learn
  defaults). Groups chain: if A~B and B~C, then A, B and C share one group.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from cvx.data.load import ID_COLUMN

logger = logging.getLogger(__name__)


def exact_duplicate_mask(df: pd.DataFrame, ignore: Sequence[str] = (ID_COLUMN,)) -> pd.Series:
    """Return True for rows that repeat an earlier row in every column except ``ignore``.

    The first copy is False (kept); later copies are True. NaN counts as equal to NaN.
    """
    columns = [c for c in df.columns if c not in ignore]
    return df.duplicated(subset=columns, keep="first")


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicates (keeping the first copy).

    Output: (DataFrame without duplicates, original index kept; number of rows removed).
    """
    mask = exact_duplicate_mask(df)
    return df.loc[~mask], int(mask.sum())


def similar_pairs(
    vectors: sparse.csr_matrix, threshold: float, batch_size: int = 500
) -> np.ndarray:
    """Find all pairs (i, j), i < j, whose cosine similarity is >= ``threshold``.

    Input: rows must already be L2-normalised (TfidfVectorizer does this), so the
    dot product equals the cosine. Rows are compared in batches to limit memory.
    Output: integer array of shape (n_pairs, 2).
    """
    vectors = sparse.csr_matrix(vectors)
    found: list[np.ndarray] = []
    for start in range(0, vectors.shape[0], batch_size):
        block = (vectors[start : start + batch_size] @ vectors.T).toarray()
        rows, cols = np.nonzero(block >= threshold)
        rows = rows + start
        keep = rows < cols  # each pair once, and not a row with itself
        found.append(np.column_stack([rows[keep], cols[keep]]))
    if not found:
        return np.empty((0, 2), dtype=int)
    return np.vstack(found).astype(int)


def groups_from_pairs(n_items: int, pairs: np.ndarray) -> np.ndarray:
    """Give every item a group id so that linked items (directly or via a chain) share it.

    Input: number of items and an (n_pairs, 2) array of linked item positions.
    Output: integer array of length ``n_items``; ids are 0, 1, 2, …
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    ones = np.ones(len(pairs), dtype=np.int8)
    graph = sparse.coo_matrix((ones, (pairs[:, 0], pairs[:, 1])), shape=(n_items, n_items))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(int)


def near_duplicate_groups(
    texts: Sequence[str], threshold: float, batch_size: int = 500
) -> np.ndarray:
    """Group texts whose TF-IDF cosine similarity is >= ``threshold``.

    Input: cleaned texts (P-CVX without labels). Output: one group id per text,
    in the same order. A text with no near-duplicate gets a group of its own.
    """
    texts = list(texts)
    if not texts:
        return np.empty(0, dtype=int)
    vectors = TfidfVectorizer().fit_transform(texts)
    pairs = similar_pairs(vectors, threshold, batch_size)
    logger.info("Near-duplicate pairs at threshold %.2f: %d", threshold, len(pairs))
    return groups_from_pairs(len(texts), pairs)


def group_summary(groups: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """Summarise near-duplicate groups.

    Output keys: rows, groups, groups_with_more_than_one_row, rows_in_such_groups,
    largest_group_size, groups_with_mixed_labels (a group holding both fake and genuine).
    """
    frame = pd.DataFrame({"group": np.asarray(groups), "label": np.asarray(labels)})
    per_group = frame.groupby("group")["label"].agg(["size", "nunique"])
    multi = per_group[per_group["size"] > 1]
    return {
        "rows": int(len(frame)),
        "groups": int(len(per_group)),
        "groups_with_more_than_one_row": int(len(multi)),
        "rows_in_such_groups": int(multi["size"].sum()),
        "largest_group_size": int(per_group["size"].max()) if len(per_group) else 0,
        "groups_with_mixed_labels": int((per_group["nunique"] > 1).sum()),
    }
