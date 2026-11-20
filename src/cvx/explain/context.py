"""Shared setup for the explanation experiments E5-E7.

They study the EXACT models exported from E4 into the model store (loaded only after
the SHA-256, version and smoke checks), on the same corrected test split. The split
settings and dataset file must match the ones recorded in the model card.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvx.artifacts.load import ModelStore, load_store
from cvx.config import Config, FeaturesConfig
from cvx.data.load import file_sha256
from cvx.data.prepare import ModelData, SplitPlan, make_model_data, make_split_plan, take


@dataclass
class XaiContext:
    """Verified models plus the full input table (text + metadata) and its split."""

    config: Config
    store: ModelStore
    plan: SplitPlan
    data: ModelData

    def test_rows(self) -> np.ndarray:
        """Positions of the test postings."""
        return self.data.split.test

    def rows(self, positions: np.ndarray) -> pd.DataFrame:
        """Model input rows at ``positions``."""
        return take(self.data.X, positions)

    def scores(self, model_id: str, positions: np.ndarray) -> np.ndarray:
        """Probability of fake from one model for the given rows."""
        estimator = self.store.models[model_id].estimator
        return np.asarray(estimator.predict_proba(self.rows(positions)))[:, 1]


def _check_card(card: dict[str, Any], config: Config, data_path: Path) -> None:
    """The run must use the same dataset file, split settings and seed as training."""
    if card.get("dataset", {}).get("sha256") != file_sha256(data_path):
        raise ValueError("The dataset differs from the one the exported models were trained on.")
    trained = card.get("split", {})
    for key, value in vars(config.split).items():
        if trained.get(key) != value:
            raise ValueError(f"split.{key} differs from the training run ({trained.get(key)!r}).")
    if card.get("seed") != config.seed:
        raise ValueError("The seed differs from the training run.")


def make_context(config: Config, postings: pd.DataFrame, data_path: Path) -> XaiContext:
    """Load the configured models and rebuild the corrected split and full input table."""
    assert config.xai is not None
    store = load_store(config.xai.model_store, config.xai.models)
    _check_card(store.card, config, data_path)
    plan = make_split_plan(postings, config)
    full = replace(config, preprocessing="p_cvx", features=FeaturesConfig(metadata=True))
    return XaiContext(config, store, plan, make_model_data(plan, full))
