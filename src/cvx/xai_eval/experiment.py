"""E6 — faithfulness, agreement, stability and runtime of SHAP vs LIME (PRD Section 8.2).

Per model: a sample of test postings (up to n/2 true positives and n/2 true negatives at
the model's ``t_f1``), the SAME sample for SHAP, LIME and the random baseline. For each
posting: comp@k, suff@k and AOPC for every method, SHAP-LIME agreement, LIME stability
over ``lime_seeds`` seeds and seconds per explanation. Then paired Wilcoxon tests on
per-posting comp@5 (SHAP vs random, LIME vs random, SHAP vs LIME).
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from cvx.evaluate.plots import save_figure
from cvx.evaluate.stats import wilcoxon_paired
from cvx.explain.context import XaiContext
from cvx.explain.core import Explanation, top_words
from cvx.explain.dispatch import explain_shap
from cvx.explain.lime_text import explain_lime, predict_texts
from cvx.features.build import TEXT_COLUMN
from cvx.runio import write_csv
from cvx.xai_eval.agreement import agreement
from cvx.xai_eval.faithfulness import eligible_words, evaluate_posting, random_rankings
from cvx.xai_eval.stability import lime_stability

METHODS = ("shap", "lime", "random")
WILCOXON_PAIRS = (("shap", "random"), ("lime", "random"), ("shap", "lime"))


def select_sample(y: np.ndarray, predicted: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Positions of up to n/2 true positives and n/2 true negatives (seeded, sorted)."""
    rng = np.random.default_rng(seed)
    picked = []
    for label in (1, 0):
        candidates = np.flatnonzero((y == label) & (predicted == label))
        size = min(n // 2, len(candidates))
        picked.append(
            rng.choice(candidates, size=size, replace=False) if size else np.empty(0, int)
        )
    return np.sort(np.concatenate(picked)).astype(int)


def _ranking(explanation: Explanation, label: int, allowed: set[str]) -> list[str]:
    """Words ordered by push towards the predicted label, kept only if removable."""
    return [w for w in top_words(explanation, len(explanation.words), toward=label) if w in allowed]


def posting_record(ctx: XaiContext, model_id: str, position: int, label: int) -> dict[str, Any]:
    """Every E6 measurement for one posting (one CSV row)."""
    loaded, xai, seed = ctx.store.models[model_id], ctx.config.xai, ctx.config.seed
    row = ctx.rows(np.array([position]))
    start = time.perf_counter()
    shap = explain_shap(loaded.estimator, row, loaded.means)[0]
    shap_seconds = time.perf_counter() - start
    limes = [
        explain_lime(loaded.estimator, row, xai.lime_samples, xai.lime_features, seed + i)
        for i in range(xai.lime_seeds)
    ]
    text = str(row[TEXT_COLUMN].iloc[0])
    words = eligible_words(text)
    rankings = {
        "shap": _ranking(shap, label, set(words)),
        "lime": _ranking(limes[0][0], label, set(words)),
    }

    def predict(texts: list[str]) -> np.ndarray:
        """p(fake) for edited texts, everything else as in the posting."""
        return predict_texts(loaded.estimator, row, texts)

    faith = evaluate_posting(
        predict,
        text,
        label,
        rankings,
        random_rankings(words, xai.random_draws, seed + position),
        xai.ks,
    )
    record: dict[str, Any] = {
        "model": model_id,
        "row_id": str(ctx.data.ids[position]),
        "label": int(ctx.data.y[position]),
        "predicted": label,
        "p_label": faith["p_label"],
        "n_words": len(words),
    }
    for method in METHODS:
        record[f"{method}_aopc"] = faith[method]["aopc"]
        for k in xai.ks:
            record[f"{method}_comp@{k}"] = faith[method]["comp"][k]
            record[f"{method}_suff@{k}"] = faith[method]["suff"][k]
    record.update(agreement(shap, limes[0][0]))
    record["lime_stability_top5"] = lime_stability([e for e, _ in limes])
    record["shap_seconds"], record["lime_seconds"] = shap_seconds, limes[0][1]
    return record


def summarise(records: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Means per method, Wilcoxon tests on comp@5, agreement, stability and runtime."""

    def column(name: str) -> np.ndarray:
        """One numeric column (None -> NaN)."""
        return np.array([np.nan if r[name] is None else r[name] for r in records], dtype=float)

    means = {
        m: {
            "aopc": float(np.nanmean(column(f"{m}_aopc"))),
            "comp": {k: float(np.nanmean(column(f"{m}_comp@{k}"))) for k in ks},
            "suff": {k: float(np.nanmean(column(f"{m}_suff@{k}"))) for k in ks},
        }
        for m in METHODS
    }
    k5 = 5 if 5 in ks else ks[0]
    tests = [
        {
            "a": a,
            "b": b,
            "metric": f"comp@{k5}",
            **wilcoxon_paired(column(f"{a}_comp@{k5}"), column(f"{b}_comp@{k5}")),
        }
        for a, b in WILCOXON_PAIRS
    ]
    agree = {
        name: float(np.nanmean(column(name)))
        for name in ("jaccard_5", "jaccard_10", "spearman", "same_direction")
    }
    runtime = {
        m: {
            "mean": float(np.mean(column(f"{m}_seconds"))),
            "p95": float(np.percentile(column(f"{m}_seconds"), 95)),
        }
        for m in ("shap", "lime")
    }
    return {
        "n_postings": len(records),
        "means": means,
        "wilcoxon": tests,
        "agreement": agree,
        "lime_stability_top5": float(np.nanmean(column("lime_stability_top5"))),
        "seconds_per_explanation": runtime,
    }


def comprehensiveness_figure(
    summary: dict[str, Any], ks: list[int], title: str, path: Path
) -> None:
    """Comprehensiveness vs k for SHAP, LIME and the random baseline."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot()
    for method in METHODS:
        ax.plot(ks, [summary["means"][method]["comp"][k] for k in ks], marker="o", label=method)
    ax.set(xlabel="k (words removed)", ylabel="mean comprehensiveness", title=title)
    ax.legend()
    save_figure(fig, path)


def run_e6(ctx: XaiContext, run_dir: Path) -> dict[str, Any]:
    """Run E6 for every configured model; write per-posting results, summary and curves."""
    xai, records_all, results = ctx.config.xai, [], {}
    for model_id, loaded in ctx.store.models.items():
        test = ctx.test_rows()
        predicted = (ctx.scores(model_id, test) >= loaded.info["thresholds"]["t_f1"]).astype(int)
        sample = select_sample(ctx.data.y[test], predicted, xai.n_postings, ctx.config.seed)
        records = [posting_record(ctx, model_id, int(test[i]), int(predicted[i])) for i in sample]
        results[model_id] = summarise(records, xai.ks)
        comprehensiveness_figure(
            results[model_id],
            xai.ks,
            f"Comprehensiveness: {model_id}",
            run_dir / "figures" / f"comprehensiveness_{model_id}.png",
        )
        records_all += records
    if records_all:
        write_csv(run_dir / "per_posting.csv", records_all, list(records_all[0]))
    return {
        "models": results,
        "settings": {
            "threshold": "t_f1",
            "ks": xai.ks,
            "lime_samples": xai.lime_samples,
            "lime_seeds": xai.lime_seeds,
            "random_draws": xai.random_draws,
        },
    }
