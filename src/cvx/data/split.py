"""Train / validation / test splits. PRD Section 6.4.

- ``random_stratified_split``: Career Verify's split (80/10/10), used by P-CV in E0/E1.
  "Stratified" means each split keeps the same fake/genuine ratio as the whole data.
- ``grouped_stratified_split``: our split (70/15/15). It is also stratified, and on top
  of that every near-duplicate group stays inside ONE split, so a posting and its
  near-copy can never be on both sides of the train/test line.

Both return positions (0..n-1), not DataFrame index labels.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Row positions belonging to each split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_labels(self, n_rows: int) -> np.ndarray:
        """Return an array of 'train' / 'val' / 'test', one entry per row position."""
        labels = np.full(n_rows, "", dtype=object)
        labels[self.train], labels[self.val], labels[self.test] = "train", "val", "test"
        return labels


def _check_ratios(ratios: tuple[float, float, float]) -> None:
    """Three positive ratios that add up to 1."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"ratios must be three positive numbers summing to 1, got {ratios}")


def random_stratified_split(
    y: np.ndarray, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> SplitIndices:
    """Stratified random split into train/val/test (Career Verify reproduction).

    Input: 0/1 labels, (train, val, test) ratios, random seed.
    Output: ``SplitIndices``. Done in two steps: cut off test, then cut val from the rest.
    """
    _check_ratios(ratios)
    y = np.asarray(y)
    positions = np.arange(len(y))
    rest, test = train_test_split(positions, test_size=ratios[2], stratify=y, random_state=seed)
    val_share_of_rest = ratios[1] / (ratios[0] + ratios[1])
    train, val = train_test_split(
        rest, test_size=val_share_of_rest, stratify=y[rest], random_state=seed
    )
    return SplitIndices(np.sort(train), np.sort(val), np.sort(test))


def _folds_per_split(ratios: tuple[float, float, float], n_folds: int) -> tuple[int, int]:
    """Number of folds for val and test; each ratio x n_folds must be a whole number."""
    counts = [r * n_folds for r in ratios]
    if any(abs(c - round(c)) > 1e-9 for c in counts):
        raise ValueError(
            f"ratios {ratios} x n_folds {n_folds} must give whole numbers of folds, got {counts}"
        )
    return round(counts[1]), round(counts[2])


def grouped_stratified_split(
    y: np.ndarray,
    groups: np.ndarray,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    n_folds: int = 20,
    seed: int = 42,
) -> SplitIndices:
    """Grouped stratified split using ``StratifiedGroupKFold``.

    The data is cut into ``n_folds`` folds that keep groups whole and the fake ratio
    similar. With 20 folds and 70/15/15: folds 0-2 -> test, 3-5 -> val, 6-19 -> train.
    Input: labels, one group id per row, ratios, fold count, seed. Output: ``SplitIndices``.
    """
    _check_ratios(ratios)
    n_val, n_test = _folds_per_split(ratios, n_folds)
    y, groups = np.asarray(y), np.asarray(groups)
    fold_of = np.full(len(y), -1)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, fold_positions) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        fold_of[fold_positions] = fold
    split = SplitIndices(
        train=np.flatnonzero(fold_of >= n_test + n_val),
        val=np.flatnonzero((fold_of >= n_test) & (fold_of < n_test + n_val)),
        test=np.flatnonzero(fold_of < n_test),
    )
    check_no_group_overlap(split, groups)
    return split


def check_no_group_overlap(split: SplitIndices, groups: np.ndarray) -> None:
    """Raise ``AssertionError`` if any group id appears in more than one split."""
    groups = np.asarray(groups)
    train, val, test = (set(groups[idx].tolist()) for idx in (split.train, split.val, split.test))
    shared = (train & val) | (train & test) | (val & test)
    if shared:
        raise AssertionError(f"{len(shared)} group(s) appear in more than one split.")
