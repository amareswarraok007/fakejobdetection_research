"""Safe loading of exported models (PRD Section 10.6; CLAUDE.md rule 9).

Loading a pickle can run code, so a model file is opened ONLY when:
1. it is listed in ``model_card.json`` inside the model store folder, and
2. its SHA-256 equals the hash recorded in the card.
Then the store is refused if the scikit-learn, XGBoost or NumPy major.minor version
differs from training, or if re-scoring the 5 smoke postings changes any score by
more than 1e-6. Any failure raises ``ModelStoreError`` with a clear message.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cvx.runio import read_json

logger = logging.getLogger(__name__)

MODEL_CARD = "model_card.json"
SMOKE_FILE = "smoke_samples.json"
CHECKED_LIBRARIES = ("scikit-learn", "xgboost", "numpy")
SMOKE_TOLERANCE = 1e-6


class ModelStoreError(RuntimeError):
    """The model store failed a safety check (missing file, hash, version or smoke score)."""


@dataclass
class LoadedModel:
    """A verified model: the pipeline, its card entry and (for linear SHAP) training means."""

    model_id: str
    estimator: Any
    info: dict[str, Any]
    means: np.ndarray | None


@dataclass
class ModelStore:
    """Every verified model plus the card and a health report for the app."""

    path: Path
    card: dict[str, Any]
    models: dict[str, LoadedModel] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


def file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_path(store: Path, card: dict[str, Any], name: str) -> Path:
    """Path of a card-listed file inside the store, after checking its SHA-256."""
    expected = card.get("sha256", {}).get(name)
    path = (store / name).resolve()
    if expected is None or path.parent != store.resolve():
        raise ModelStoreError(f"'{name}' is not listed in {MODEL_CARD}; refusing to load it.")
    if not path.is_file():
        raise ModelStoreError(f"'{name}' is listed in {MODEL_CARD} but missing from {store}.")
    if file_hash(path) != expected:
        raise ModelStoreError(f"SHA-256 of '{name}' does not match {MODEL_CARD}; refusing to load.")
    return path


def _major_minor(text: str | None) -> str | None:
    """'1.6.1' -> '1.6'."""
    return None if text is None else ".".join(text.split(".")[:2])


def version_problems(card: dict[str, Any]) -> list[str]:
    """Libraries whose installed major.minor version differs from the training run."""
    trained = card.get("libraries", {})
    problems = []
    for name in CHECKED_LIBRARIES:
        installed = version(name)
        if _major_minor(trained.get(name)) != _major_minor(installed):
            problems.append(f"{name}: trained with {trained.get(name)}, installed {installed}")
    return problems


def load_model(store: Path, card: dict[str, Any], model_id: str) -> LoadedModel:
    """Verify and load one model's files listed in the card."""
    info = card.get("models", {}).get(model_id)
    if info is None:
        raise ModelStoreError(f"Model '{model_id}' is not in {MODEL_CARD}.")
    files = info["files"]
    estimator = joblib.load(verified_path(store, card, files["model"]))
    means = None
    if "feature_means" in files:
        means = np.load(verified_path(store, card, files["feature_means"]), allow_pickle=False)
    for name in files.values():  # every other listed file must also be intact
        verified_path(store, card, name)
    return LoadedModel(model_id, estimator, info, means)


def smoke_differences(store: ModelStore, smoke: dict[str, Any]) -> dict[str, float]:
    """Largest |new score - stored score| per model on the smoke postings."""
    rows = pd.DataFrame(smoke["rows"])
    differences = {}
    for model_id, loaded in store.models.items():
        new = np.asarray(loaded.estimator.predict_proba(rows))[:, 1]
        differences[model_id] = float(np.abs(new - np.asarray(smoke["scores"][model_id])).max())
    return differences


def load_store(store_dir: str | Path, model_ids: list[str] | None = None) -> ModelStore:
    """Load and check the model store (all models in the card, or ``model_ids``)."""
    path = Path(store_dir)
    if not (path / MODEL_CARD).is_file():
        raise ModelStoreError(
            f"No {MODEL_CARD} in '{path}'. Export models first: "
            "python scripts/export_model.py --run runs/e4_ablations/<timestamp> ..."
        )
    card = read_json(path / MODEL_CARD)
    problems = version_problems(card)
    if problems:
        raise ModelStoreError("Library versions differ from training: " + "; ".join(problems))
    store = ModelStore(path, card)
    for model_id in model_ids or list(card["models"]):
        store.models[model_id] = load_model(path, card, model_id)
    differences = smoke_differences(store, read_json(verified_path(path, card, SMOKE_FILE)))
    bad = {m: d for m, d in differences.items() if d > SMOKE_TOLERANCE}
    if bad:
        raise ModelStoreError(f"Smoke scores changed by more than {SMOKE_TOLERANCE}: {bad}")
    store.report = {
        "status": "ok",
        "models": list(store.models),
        "smoke_max_difference": differences,
        "installed": {name: version(name) for name in CHECKED_LIBRARIES},
        "trained": {name: card.get("libraries", {}).get(name) for name in CHECKED_LIBRARIES},
        "source_run": card.get("source_run"),
    }
    logger.info("Model store OK: %s", list(store.models))
    return store
