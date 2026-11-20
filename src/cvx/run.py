"""Run one experiment from its config:  python -m cvx.run --config configs/e0_reproduce.yaml

For every variant (or the single "main" variant) and every model in the config:
prepare data -> tune -> fit on train -> thresholds on validation -> score the test
split ONCE -> metrics (+ bootstrap CIs). Then McNemar tests and, if the config names a
``compare_to`` variant, the "reported vs realistic" comparison (E1). All results are
written to a new run folder (see ``runio.py``); nothing is typed by hand.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cvx import runio
from cvx.config import Config, expand_variants, load_config
from cvx.data.load import load_postings
from cvx.data.prepare import ModelData, make_model_data, make_split_plan, split_key, take
from cvx.data.resample import apply_before_split
from cvx.evaluate.metrics import evaluate_scores, split_summary
from cvx.evaluate.plots import plot_pr_curves, plot_roc_curves
from cvx.evaluate.stats import bootstrap_ci, mcnemar_exact
from cvx.models.train import (
    FoldData,
    cv_folds,
    make_head_factory,
    predict_scores,
    resolve_thresholds,
    train_model,
    train_val_fold,
)
from cvx.models.zoo import candidates, grid_for
from cvx.tasks import run_task

logger = logging.getLogger(__name__)
CV_COLUMNS = ["variant", "model", "params", "fold", "scoring", "score"]
PREDICTION_COLUMNS = ["variant", "model", "row_id", "label", "score"]


@dataclass
class TestRecord:
    """Test-set labels and scores of one fitted model (for McNemar and comparisons)."""

    ids: np.ndarray
    y: np.ndarray
    scores: np.ndarray
    thresholds: dict[str, float]


@dataclass
class Collected:
    """Everything gathered while running the variants."""

    variants: dict[str, Any] = field(default_factory=dict)
    cv_rows: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    records: dict[str, TestRecord] = field(default_factory=dict)


class _DataCache:
    """Share split plans, model inputs and fitted fold features between variants.

    These do not depend on the model, so variants with the same data and feature
    settings (e.g. E4 'text' and 'text_smote') reuse them instead of rebuilding.
    """

    def __init__(self, postings: pd.DataFrame) -> None:
        """Start empty; ``postings`` is the full loaded dataset."""
        self.postings = postings
        self.plans: dict[Any, Any] = {}
        self.inputs: dict[Any, ModelData] = {}
        self.folds: dict[Any, tuple[FoldData, list[FoldData] | None]] = {}

    def _input_key(self, config: Config) -> tuple[Any, ...]:
        """Settings that decide the model input table."""
        features = config.features
        return (
            split_key(config),
            config.preprocessing,
            features.metadata,
            tuple(features.extra_categories),
        )

    def get(self, config: Config) -> ModelData:
        """Model data for ``config`` (before-split resampling applied, if any)."""
        key = split_key(config)
        if key not in self.plans:
            self.plans[key] = make_split_plan(self.postings, config)
        input_key = self._input_key(config)
        if input_key not in self.inputs:
            self.inputs[input_key] = make_model_data(self.plans[key], config)
        return apply_before_split(self.inputs[input_key], config)

    def get_folds(
        self, config: Config, data: ModelData, need_cv: bool
    ) -> tuple[FoldData, list[FoldData] | None]:
        """Train/validation features and (if ``need_cv``) the grouped CV fold features."""
        before_split = config.resample is not None and config.resample.when == "before_split"
        key = (*self._input_key(config), repr(config.tfidf), repr(config.tuning), need_cv)
        if before_split or key not in self.folds:
            folds = (train_val_fold(data, config), cv_folds(data, config) if need_cv else None)
            if before_split:  # resampled data is rebuilt for each variant; do not share
                return folds
            self.folds[key] = folds
        return self.folds[key]


def run_model(
    model_id: str,
    config: Config,
    data: ModelData,
    folds: tuple[FoldData, list[FoldData] | None],
) -> tuple[dict[str, Any], Any, TestRecord]:
    """Tune and fit one model, then score validation and (once) the test split."""
    grid_candidates = candidates(grid_for(model_id, config.grids))
    train_val, cv = folds
    trained = train_model(
        model_id,
        make_head_factory(model_id, config),
        grid_candidates,
        cv,
        train_val,
        config.tuning.for_model(model_id),
    )
    thresholds = resolve_thresholds(config.thresholds or [0.5], trained.thresholds)
    test = data.split.test
    record = TestRecord(
        data.ids[test],
        data.y[test],
        predict_scores(trained.estimator, take(data.X, test)),
        thresholds,
    )
    result = {
        "best_params": trained.best_params,
        "tuning": {**vars(config.tuning.for_model(model_id)), "best_score": trained.best_score},
        "chosen_thresholds": trained.thresholds,
        "fit_seconds": trained.seconds,
        "validation": evaluate_scores(data.y[data.split.val], trained.val_scores, thresholds),
        "test": evaluate_scores(record.y, record.scores, thresholds),
    }
    if config.stats is not None and config.stats.bootstrap:
        result["test_bootstrap_ci"] = bootstrap_ci(
            record.y, record.scores, thresholds, config.stats.bootstrap, config.seed
        )
    return result, trained, record


def _variant_summary(config: Config, data: ModelData) -> dict[str, Any]:
    """Settings that identify a variant, plus its split sizes and fake shares."""
    return {
        "preprocessing": config.preprocessing,
        "split_method": config.split.method,
        "features": vars(config.features),
        "resample": None if config.resample is None else vars(config.resample),
        "prevectorized": data.prevectorized,
        "split_summary": split_summary(data.y, data.split),
        "models": {},
    }


def run_variant(
    name: str, config: Config, cache: _DataCache, run_dir: Path, out: Collected
) -> None:
    """Run every model of one variant and add its results to ``out``."""
    logger.info("Variant %s", name)
    data = cache.get(config)
    need_cv = any(
        config.tuning.for_model(m).method == "grouped_cv"
        and len(candidates(grid_for(m, config.grids))) > 1
        for m in config.models
    )
    folds = cache.get_folds(config, data, need_cv)
    summary = _variant_summary(config, data)
    curves = {}
    for model_id in config.models:
        result, trained, record = run_model(model_id, config, data, folds)
        summary["models"][model_id] = result
        out.records[f"{name}/{model_id}"] = record
        curves[model_id] = (record.y, record.scores)
        out.cv_rows += [
            {
                "variant": name,
                "model": model_id,
                "params": json.dumps(r["params"], sort_keys=True),
                "fold": r["fold"],
                "scoring": config.tuning.scoring,
                "score": r["score"],
            }
            for r in trained.cv_rows
        ]
        out.predictions += [
            {"variant": name, "model": model_id, "row_id": i, "label": int(t), "score": float(s)}
            for i, t, s in zip(record.ids, record.y, record.scores, strict=True)
        ]
        if config.export_models:
            runio.save_model(run_dir, f"{model_id}_{name}", trained.estimator)
    out.variants[name] = summary
    plot_pr_curves(curves, run_dir / "figures" / f"pr_{name}.png", f"PR curve: {name}")
    plot_roc_curves(curves, run_dir / "figures" / f"roc_{name}.png", f"ROC curve: {name}")


def _resolve_key(item: str, variant_names: list[str]) -> str:
    """'lr' -> 'main/lr' when there is one variant; 'variant/model' is kept as is."""
    if "/" in item:
        return item
    if len(variant_names) != 1:
        raise ValueError(f"McNemar item {item!r} must be 'variant/model' (several variants).")
    return f"{variant_names[0]}/{item}"


def mcnemar_tests(config: Config, out: Collected) -> list[dict[str, Any]]:
    """McNemar's test for each configured pair, at each threshold name."""
    if config.stats is None:
        return []
    names, tests = list(out.variants), []
    for pair in config.stats.mcnemar:
        key_a, key_b = (_resolve_key(item, names) for item in pair)
        a, b = out.records[key_a], out.records[key_b]
        if not np.array_equal(a.ids, b.ids):
            raise ValueError(f"McNemar needs the same test postings: {key_a} vs {key_b}.")
        for t_name in a.thresholds:
            result = mcnemar_exact(
                a.y, a.scores >= a.thresholds[t_name], b.scores >= b.thresholds[t_name]
            )
            tests.append({"model_a": key_a, "model_b": key_b, "threshold": t_name, **result})
    return tests


def _pair(reported: float | None, realistic: float | None) -> dict[str, float | None]:
    """Reported value, realistic value and their difference (reported - realistic)."""
    diff = None if reported is None or realistic is None else reported - realistic
    return {"reported": reported, "realistic": realistic, "difference": diff}


def reported_vs_realistic(reference: str, out: Collected) -> list[dict[str, Any]]:
    """E1: compare each variant with the ``compare_to`` (realistic) variant, model by model.

    PR-AUC, ROC-AUC and fake-class recall at each threshold, plus the test fake share,
    because the test prevalence differs between the two settings.
    """
    ref = out.variants[reference]
    rows = []
    for name, variant in out.variants.items():
        if name == reference:
            continue
        for model_id, result in variant["models"].items():
            ref_result = ref["models"].get(model_id)
            if ref_result is None:
                continue
            new, old = result["test"], ref_result["test"]
            rows.append(
                {
                    "reported": f"{name}/{model_id}",
                    "realistic": f"{reference}/{model_id}",
                    "test_fake_share": _pair(
                        variant["split_summary"]["test"]["fake_share"],
                        ref["split_summary"]["test"]["fake_share"],
                    ),
                    "pr_auc": _pair(
                        new["threshold_free"]["pr_auc"], old["threshold_free"]["pr_auc"]
                    ),
                    "roc_auc": _pair(
                        new["threshold_free"]["roc_auc"], old["threshold_free"]["roc_auc"]
                    ),
                    "recall": {
                        t: _pair(new["at_threshold"][t]["recall"], old["at_threshold"][t]["recall"])
                        for t in new["at_threshold"]
                    },
                }
            )
    return rows


def run_training(config: Config, postings: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    """E0-E4: every variant and model; writes cv_results.csv and predictions.csv."""
    cache, out = _DataCache(postings), Collected()
    for name, variant_config in expand_variants(config):
        run_variant(name, variant_config, cache, run_dir, out)
    metrics: dict[str, Any] = {"variants": out.variants, "mcnemar": mcnemar_tests(config, out)}
    if config.compare_to:
        metrics["reported_vs_realistic"] = reported_vs_realistic(config.compare_to, out)
    runio.write_csv(run_dir / "cv_results.csv", out.cv_rows, CV_COLUMNS)
    runio.write_csv(run_dir / "predictions.csv", out.predictions, PREDICTION_COLUMNS)
    return metrics


def run_experiment(
    config_path: str | Path, data_path: str | Path | None = None, runs_dir: str | Path | None = None
) -> Path:
    """Run one config (training E0-E4, or a task E5-E8) and write a complete run folder."""
    config = load_config(config_path)
    if config.task == "train" and not config.models:
        raise ValueError(f"{config_path} lists no models; nothing to run.")
    data_file = Path(data_path or config.paths.data_csv)
    run_dir = runio.create_run_dir(runs_dir or config.paths.runs_dir, config.experiment_id)
    handler = runio.attach_log_file(run_dir)
    try:
        runio.write_config(run_dir, config)
        env = runio.start_env(run_dir, config, data_file)
        postings = load_postings(data_file)
        if config.task == "train":
            results = run_training(config, postings, run_dir)
        else:
            results = run_task(config, postings, data_file, run_dir)
        metrics = {
            "experiment_id": config.experiment_id,
            "run_id": run_dir.name,
            "task": config.task,
            **results,
        }
        runio.write_json(run_dir / "metrics.json", metrics)
        runio.finish_env(run_dir, env)
        logger.info("Run complete: %s", run_dir)
    finally:
        runio.detach_log_file(handler)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Run one CVX experiment config.")
    parser.add_argument(
        "--config", required=True, help="YAML config, e.g. configs/e0_reproduce.yaml"
    )
    parser.add_argument("--data", default=None, help="CSV path (default: paths.data_csv)")
    parser.add_argument("--runs-dir", default=None, help="Output root (default: paths.runs_dir)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_dir = run_experiment(args.config, args.data, args.runs_dir)
    logger.info("Run folder: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
