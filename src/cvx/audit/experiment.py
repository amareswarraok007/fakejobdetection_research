"""E7 — shortcut audit and error analysis (PRD Section 8.2).

For each configured model (the text-only models by default):
- baseline: the exported model's test metrics, global top-30 SHAP features, flagged tokens;
- E7a: re-train the same model type (same hyperparameters, from the model card) with the
  audit columns (location, industry) added; report the metric change and the share of the
  new model's top-30 features that come from those columns;
- E7b: re-train with the flagged tokens replaced by ``maskedtoken``; report the change;
- errors.csv: 20 false negatives and 20 false positives with their top-5 SHAP features.
Re-training uses the training split only; thresholds come from validation; test once.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from cvx.artifacts.save import parse_model_id
from cvx.audit.errors import ERROR_COLUMNS, error_rows
from cvx.audit.shortcuts import audit_share, flag_tokens, mask_text, place_words
from cvx.config import Config, FeaturesConfig
from cvx.data.clean import stop_words
from cvx.data.prepare import ModelData, make_model_data, take
from cvx.evaluate.metrics import evaluate_scores
from cvx.explain.context import XaiContext
from cvx.explain.core import feature_names, global_ranking
from cvx.explain.dispatch import shap_matrix
from cvx.explain.experiment import global_top
from cvx.explain.shap_linear import training_means
from cvx.features.build import TEXT_COLUMN
from cvx.models.train import make_head_factory, predict_scores, train_model, train_val_fold
from cvx.runio import write_csv

COMPARED = ("pr_auc", "roc_auc")


def _model_config(ctx: XaiContext, model_id: str, extra: tuple[str, ...]) -> Config:
    """The experiment config with the model's feature set (+ optional audit columns)."""
    info = ctx.store.models[model_id].info
    return replace(
        ctx.config,
        preprocessing="p_cvx",
        features=FeaturesConfig(metadata=bool(info["metadata"]), extra_categories=list(extra)),
    )


def retrain(ctx: XaiContext, model_id: str, config: Config, data: ModelData) -> dict[str, Any]:
    """Fit the model type with its card hyperparameters; return test metrics and top-30."""
    model, _ = parse_model_id(model_id)
    params = ctx.store.models[model_id].info["best_params"]
    trained = train_model(
        model,
        make_head_factory(model, config),
        [params],
        None,
        train_val_fold(data, config),
        config.tuning,
    )
    test = data.split.test
    scores = predict_scores(trained.estimator, take(data.X, test))
    thresholds = {"0.5": 0.5, **trained.thresholds}
    means = training_means(trained.estimator, take(data.X, data.split.train))
    rows = take(data.X, test[: ctx.config.xai.n_global])
    phi = shap_matrix(trained.estimator, rows, means)
    ranking = global_ranking(feature_names(trained.estimator), phi, ctx.config.xai.top_global)
    return {
        "test": evaluate_scores(data.y[test], scores, thresholds),
        "global_top": ranking,
        "audit_share_top": audit_share(ranking),
    }


def _change(new: dict[str, Any], base: dict[str, Any]) -> dict[str, float | None]:
    """new - base for PR-AUC, ROC-AUC and F1 at t_f1."""
    change = {}
    for metric in COMPARED:
        a, b = new["threshold_free"][metric], base["threshold_free"][metric]
        change[metric] = None if a is None or b is None else a - b
    change["f1@t_f1"] = new["at_threshold"]["t_f1"]["f1"] - base["at_threshold"]["t_f1"]["f1"]
    return change


def audit_model(ctx: XaiContext, model_id: str, places: set[str]) -> dict[str, Any]:
    """Baseline, E7a and E7b for one exported model."""
    loaded, xai = ctx.store.models[model_id], ctx.config.xai
    test = ctx.test_rows()
    base = evaluate_scores(
        ctx.data.y[test],
        ctx.scores(model_id, test),
        {"0.5": 0.5, **{k: v for k, v in loaded.info["thresholds"].items()}},
    )
    ranking = global_top(ctx, model_id)
    flagged = flag_tokens(ranking, places, xai.mask_tokens)
    config_a = _model_config(ctx, model_id, tuple(xai.audit_columns))
    e7a = retrain(ctx, model_id, config_a, make_model_data(ctx.plan, config_a))
    config_b = _model_config(ctx, model_id, ())
    data_b = make_model_data(ctx.plan, config_b)
    masked = data_b.X.assign(**{TEXT_COLUMN: mask_text(data_b.X[TEXT_COLUMN], flagged)})
    e7b = retrain(ctx, model_id, config_b, replace(data_b, X=masked))
    return {
        "baseline": {"test": base, "global_top": ranking},
        "flagged_tokens": flagged,
        "e7a_audit_columns": {
            **e7a,
            "columns": list(xai.audit_columns),
            "change": _change(e7a["test"], base),
        },
        "e7b_masked": {**e7b, "masked_tokens": flagged, "change": _change(e7b["test"], base)},
    }


def run_e7(ctx: XaiContext, run_dir: Path) -> dict[str, Any]:
    """Run E7 for every configured model; write errors.csv."""
    places = place_words(ctx.plan.postings, stop_words())
    results, errors = {}, []
    for model_id in ctx.store.models:
        results[model_id] = audit_model(ctx, model_id, places)
        errors += error_rows(ctx, model_id, ctx.config.xai.n_errors)
    write_csv(run_dir / "errors.csv", errors, ERROR_COLUMNS)
    return {"models": results, "note": "Group errors.csv by cause by hand (PRD E7)."}
