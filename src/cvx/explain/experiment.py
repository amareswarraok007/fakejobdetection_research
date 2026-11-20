"""E5 — explanation generation and correctness checks (PRD Section 8.2).

For every exported model:
- check: linear SHAP formula vs ``shap.LinearExplainer`` (max difference, must be < 1e-6)
  or Tree SHAP additivity (base + sum = margin, must be < 1e-4), on ``n_check`` test postings;
- global ranking: mean |SHAP| over up to ``n_global`` test postings, top ``top_global`` saved;
- examples: SHAP + LIME explanations and reasons for ``n_examples`` test postings.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from cvx.evaluate.plots import save_figure, shap_bar_figure
from cvx.explain.context import XaiContext
from cvx.explain.core import feature_names, global_ranking, top_words
from cvx.explain.dispatch import explain_shap, model_kind, shap_matrix
from cvx.explain.lime_text import explain_lime
from cvx.explain.reasons import reasons
from cvx.explain.shap_linear import check_against_shap
from cvx.explain.shap_tree import additivity_error
from cvx.runio import write_csv
from cvx.xai_eval.agreement import jaccard

CHECK_LIMITS = {"linear": 1e-6, "tree": 1e-4}
CHUNK = 200  # rows per SHAP batch, to keep memory small


def correctness_check(ctx: XaiContext, model_id: str) -> dict[str, Any]:
    """The PRD check for this model type on the first ``n_check`` test postings."""
    loaded = ctx.store.models[model_id]
    rows = ctx.rows(ctx.test_rows()[: ctx.config.xai.n_check])
    kind = model_kind(loaded.estimator)
    if kind == "linear":
        value, name = (
            check_against_shap(loaded.estimator, rows, loaded.means),
            "max |formula - LinearExplainer|",
        )
    else:
        value, name = additivity_error(loaded.estimator, rows), "max |base + sum(SHAP) - margin|"
    return {
        "kind": kind,
        "check": name,
        "value": value,
        "limit": CHECK_LIMITS[kind],
        "passed": bool(value < CHECK_LIMITS[kind]),
        "n_postings": int(len(rows)),
    }


def mean_abs_shap(ctx: XaiContext, model_id: str, positions: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature over ``positions`` (computed in small batches)."""
    loaded = ctx.store.models[model_id]
    total = None
    for start in range(0, len(positions), CHUNK):
        phi = np.abs(
            shap_matrix(loaded.estimator, ctx.rows(positions[start : start + CHUNK]), loaded.means)
        )
        total = phi.sum(axis=0) if total is None else total + phi.sum(axis=0)
    return total / len(positions)


def global_top(ctx: XaiContext, model_id: str) -> list[dict[str, Any]]:
    """Top ``top_global`` features by mean |SHAP| over the first ``n_global`` test postings."""
    positions = ctx.test_rows()[: ctx.config.xai.n_global]
    names = feature_names(ctx.store.models[model_id].estimator)
    return global_ranking(
        names, mean_abs_shap(ctx, model_id, positions)[None, :], ctx.config.xai.top_global
    )


def example(ctx: XaiContext, model_id: str, position: int) -> dict[str, Any]:
    """SHAP and LIME explanations, reasons and their top-5 overlap for one test posting."""
    loaded, xai = ctx.store.models[model_id], ctx.config.xai
    row = ctx.rows(np.array([position]))
    shap = explain_shap(loaded.estimator, row, loaded.means)[0]
    lime, seconds = explain_lime(
        loaded.estimator, row, xai.lime_samples, xai.lime_features, ctx.config.seed
    )
    return {
        "model": model_id,
        "row_id": str(ctx.data.ids[position]),
        "label": int(ctx.data.y[position]),
        "score": shap.score,
        "shap_top10": {w: shap.words[w] for w in top_words(shap, 10)},
        "lime_top10": {w: lime.words[w] for w in top_words(lime, 10)},
        "jaccard_top5": jaccard(top_words(shap, 5), top_words(lime, 5)),
        "reasons": reasons(shap, row),
        "lime_seconds": seconds,
    }


def run_e5(ctx: XaiContext, run_dir: Path) -> dict[str, Any]:
    """Run E5 for every configured model; write global rankings, examples and figures."""
    results: dict[str, Any] = {}
    examples = []
    for model_id in ctx.store.models:
        ranking = global_top(ctx, model_id)
        write_csv(run_dir / f"global_shap_{model_id}.csv", ranking, ["feature", "mean_abs_shap"])
        figure = shap_bar_figure(
            [(r["feature"], r["mean_abs_shap"]) for r in ranking], f"Global mean |SHAP|: {model_id}"
        )
        save_figure(figure, run_dir / "figures" / f"global_shap_{model_id}.png")
        results[model_id] = {"check": correctness_check(ctx, model_id), "global_top": ranking}
        positions = ctx.test_rows()[: ctx.config.xai.n_examples]
        examples += [example(ctx, model_id, int(p)) for p in positions]
    lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in examples)
    (run_dir / "explanations.jsonl").write_text(lines + "\n", encoding="utf-8")
    return {
        "models": results,
        "all_checks_passed": all(r["check"]["passed"] for r in results.values()),
    }
