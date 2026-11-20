"""Export chosen models from a run folder into the model store (PRD Section 10.6).

Files written for each model id (e.g. ``lr_full`` = Logistic Regression, variant 'full'):
``<id>.pkl`` (the whole pipeline), ``<id>_features.json`` (feature names),
``<id>_feature_means.npy`` (training means, for exact linear SHAP),
``<id>_booster.json`` (XGBoost backup); plus ``smoke_samples.json`` (5 fixed test
postings with every model's score) and ``model_card.json`` (hashes, thresholds,
test metrics copied from the run, library versions, dataset hash, seed, git commit).

The run's model file is COPIED byte for byte; it is only unpickled after its SHA-256
has been written to the card and checked (``load.verified_path``).
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from cvx.artifacts.load import MODEL_CARD, SMOKE_FILE, file_hash, load_model
from cvx.config import MODEL_IDS, Config, expand_variants, load_config
from cvx.data.load import file_sha256, load_postings
from cvx.data.prepare import SplitPlan, make_model_data, make_split_plan, take
from cvx.explain.core import feature_names
from cvx.explain.dispatch import model_kind
from cvx.explain.shap_linear import training_means
from cvx.runio import STATUS_COMPLETE, read_json, utc_now, write_json

logger = logging.getLogger(__name__)
N_SMOKE = 5


def parse_model_id(model_id: str) -> tuple[str, str]:
    """'linear_svm_text' -> ('linear_svm', 'text'); 'lr_full' -> ('lr', 'full')."""
    for model in sorted(MODEL_IDS, key=len, reverse=True):
        if model_id.startswith(model + "_"):
            return model, model_id[len(model) + 1 :]
    raise ValueError(f"Model id {model_id!r} must look like '<model>_<variant>', e.g. 'lr_full'.")


def _check_run(run_dir: Path, data_path: Path) -> dict[str, Any]:
    """The run must be complete and made from the same dataset file."""
    env = read_json(run_dir / "env.json")
    if env.get("status") != STATUS_COMPLETE:
        raise ValueError(f"{run_dir} is not a complete run (env.json status).")
    if env["dataset"]["sha256"] != file_sha256(data_path):
        raise ValueError(f"{data_path} is not the dataset used by {run_dir}.")
    return env


def _base_card(run_dir: Path, config: Config, env: dict[str, Any]) -> dict[str, Any]:
    """Card fields shared by all models."""
    return {
        "created_at_utc": utc_now(),
        "source_run": f"{run_dir.parent.name}/{run_dir.name}",
        "git": env.get("git"),
        "dataset": env.get("dataset"),
        "seed": config.seed,
        "split": {**vars(config.split)},
        "libraries": env.get("libraries", {}),
        "models": {},
        "sha256": {},
    }


def _write_card(out: Path, card: dict[str, Any]) -> None:
    """Refresh every file hash and write ``model_card.json``."""
    for name in list(card["sha256"]):
        card["sha256"][name] = file_hash(out / name)
    write_json(out / MODEL_CARD, card)


def _export_one(
    model_id: str,
    out: Path,
    card: dict[str, Any],
    variants: dict[str, Config],
    metrics: dict[str, Any],
    plan: SplitPlan,
) -> None:
    """Add one (already copied and hashed) model's extra files and card entry."""
    model, variant = parse_model_id(model_id)
    config = variants[variant]
    if config.preprocessing != "p_cvx":
        raise ValueError(f"{model_id}: the app uses P-CVX models only.")
    loaded = load_model(out, card, model_id)  # SHA-256 checked before unpickling
    data = make_model_data(plan, config)
    files = card["models"][model_id]["files"]
    files["features"] = f"{model_id}_features.json"
    write_json(out / files["features"], feature_names(loaded.estimator).tolist())
    files["feature_means"] = f"{model_id}_feature_means.npy"
    np.save(
        out / files["feature_means"],
        training_means(loaded.estimator, take(data.X, data.split.train)),
    )
    kind = model_kind(loaded.estimator)
    if kind == "tree":
        files["booster"] = f"{model_id}_booster.json"
        loaded.estimator.named_steps["clf"].save_model(out / files["booster"])
    result = metrics["variants"][variant]["models"][model]
    card["models"][model_id].update(
        model=model,
        variant=variant,
        kind=kind,
        metadata=config.features.metadata,
        preprocessing=config.preprocessing,
        input_columns=list(data.X.columns),
        best_params=result["best_params"],
        thresholds=result["chosen_thresholds"],
        test_metrics=result["test"],
    )
    for name in files.values():
        card["sha256"][name] = ""


def _smoke_samples(
    out: Path, card: dict[str, Any], plan: SplitPlan, variants: dict[str, Config]
) -> None:
    """Five fixed test postings (full input table) with every exported model's score."""
    full = (
        next(c for c in variants.values() if c.features.metadata)
        if any(c.features.metadata for c in variants.values())
        else next(iter(variants.values()))
    )
    data = make_model_data(plan, full)
    rows = take(data.X, data.split.test[:N_SMOKE])
    scores = {}
    for model_id in card["models"]:
        estimator = load_model(out, card, model_id).estimator
        scores[model_id] = np.asarray(estimator.predict_proba(rows))[:, 1].tolist()
    write_json(
        out / SMOKE_FILE,
        {
            "ids": data.ids[data.split.test[:N_SMOKE]].tolist(),
            "rows": rows.to_dict(orient="records"),
            "scores": scores,
        },
    )
    card["sha256"][SMOKE_FILE] = ""


def export_models(
    run_dir: str | Path, model_ids: list[str], out_dir: str | Path, data_path: str | Path
) -> Path:
    """Export ``model_ids`` from a complete run into ``out_dir``. Output: card path."""
    run, out, data_file = Path(run_dir), Path(out_dir), Path(data_path)
    env = _check_run(run, data_file)
    config = load_config(run / "config.yaml")
    metrics = read_json(run / "metrics.json")
    variants = dict(expand_variants(config))
    out.mkdir(parents=True, exist_ok=True)
    card = _base_card(run, config, env)
    for model_id in model_ids:
        source = run / "models" / f"{model_id}.joblib"
        if not source.is_file():
            raise FileNotFoundError(f"{source} not found (did the config set export_models?).")
        shutil.copyfile(source, out / f"{model_id}.pkl")
        card["models"][model_id] = {"files": {"model": f"{model_id}.pkl"}}
        card["sha256"][f"{model_id}.pkl"] = ""
    _write_card(out, card)
    plan = make_split_plan(load_postings(data_file), next(iter(variants.values())))
    for model_id in model_ids:
        _export_one(model_id, out, card, variants, metrics, plan)
        _write_card(out, card)
    _smoke_samples(out, card, plan, variants)
    _write_card(out, card)
    logger.info("Exported %s to %s", model_ids, out)
    return out / MODEL_CARD
