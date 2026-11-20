"""Class balancing for E1 (PRD Section 8.2) and the E4 SMOTE ablation.

- ``before_split`` (E1a, E1b): the WHOLE dataset is balanced to 1:1 and only then split.
  Copies or SMOTE blends of a fake posting can land in both train and test, and the
  test set becomes about half fake. This imitates the likely Career Verify setup.
  E1b vectorises the text with TF-IDF on all postings first, because SMOTE needs numbers.
- ``train_only`` (E1c control, E4): a sampler step inside the model pipeline, so only
  training rows are resampled; validation and test keep their natural prevalence.
"""

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy import sparse

from cvx.config import Config
from cvx.data.prepare import ModelData, with_rows
from cvx.data.split import random_stratified_split
from cvx.features.build import TEXT_COLUMN
from cvx.features.text import make_tfidf

SMOTE_NEIGHBOURS = 5


def make_sampler(method: str, seed: int) -> RandomOverSampler | SMOTE:
    """Create an imbalanced-learn sampler that balances classes 1:1."""
    if method == "random_oversample":
        return RandomOverSampler(random_state=seed)
    if method == "smote":
        return SMOTE(k_neighbors=SMOTE_NEIGHBOURS, random_state=seed)
    raise ValueError(f"Unknown resampling method {method!r}")


def oversample_before_split(data: ModelData, ratios: tuple[float, ...], seed: int) -> ModelData:
    """E1a: duplicate random fake rows until 1:1, then a stratified random split."""
    positions = np.arange(len(data.y)).reshape(-1, 1)
    resampled, _ = RandomOverSampler(random_state=seed).fit_resample(positions, data.y)
    positions = resampled.ravel()
    new_split = random_stratified_split(data.y[positions], ratios, seed)
    return with_rows(data, positions, new_split)


def smote_before_split(
    data: ModelData, config: Config, ratios: tuple[float, ...], seed: int
) -> ModelData:
    """E1b: TF-IDF on all postings, SMOTE to 1:1, then a stratified random split.

    SMOTE-made rows get ids ``smote_0``, ``smote_1``, … so they are easy to spot.
    The result is prevectorized: models are trained on this matrix directly.
    """
    matrix = make_tfidf(config.tfidf).fit_transform(data.X[TEXT_COLUMN])
    X_res, y_res = make_sampler("smote", seed).fit_resample(matrix, data.y)
    n_new = len(y_res) - len(data.y)
    ids = np.concatenate([data.ids, np.array([f"smote_{i}" for i in range(n_new)])])
    new_split = random_stratified_split(y_res, ratios, seed)
    return ModelData(
        X=sparse.csr_matrix(X_res),
        y=np.asarray(y_res).astype(int),
        ids=ids,
        split=new_split,
        groups=None,
        prevectorized=True,
    )


def apply_before_split(data: ModelData, config: Config) -> ModelData:
    """Apply the config's before-split resampling, or return ``data`` unchanged."""
    resample = config.resample
    if resample is None or resample.when != "before_split":
        return data
    ratios = tuple(config.split.ratios)
    if resample.method == "random_oversample":
        return oversample_before_split(data, ratios, config.seed)
    return smote_before_split(data, config, ratios, config.seed)
