"""Markdown result tables, built only from run folders (and the owner's paper-values file).

Every number printed here is read from a ``metrics.json`` written by ``cvx.run``,
except the Career Verify column, which is read from ``docs/career_verify_reported.yaml``
filled in by the owner from the paper. Only CVX run folders are read, so results
from other notebooks (for example the earlier DistilBERT notebook, which used a
different split) can never enter these tables or McNemar tests (PRD §9.1).
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from cvx.report.runs import RunData, latest_runs

logger = logging.getLogger(__name__)
MISSING = "n/a"
PAPER_TODO = "<<fill docs/career_verify_reported.yaml>>"
PAPER_FILL = "FILL"  # placeholder value in the template, replaced by the owner
PAPER_THRESHOLD = "0.5"  # the paper's plain predict() rule


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table: floats to ``digits`` places, None as 'n/a'."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def with_ci(value: Any, ci: dict[str, Any] | None) -> str:
    """'0.812 [0.770, 0.850]' when a CI exists, else just the value."""
    if not ci or ci.get("low") is None:
        return fmt(value)
    return f"{fmt(value)} [{fmt(ci['low'])}, {fmt(ci['high'])}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def model_results(
    metrics: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Yield (variant, model, model result, variant summary) for every fitted model."""
    for variant, summary in metrics["variants"].items():
        for model, result in summary["models"].items():
            yield variant, model, result, summary


def score_table(metrics: dict[str, Any]) -> str:
    """Threshold-free test metrics with bootstrap CIs, one row per variant/model."""
    rows = []
    for variant, model, result, summary in model_results(metrics):
        test, ci = result["test"]["threshold_free"], result.get("test_bootstrap_ci", {})
        split_test = summary["split_summary"]["test"]
        rows.append(
            [
                variant,
                model,
                fmt(split_test["rows"]),
                fmt(split_test["fake_share"]),
                with_ci(test["pr_auc"], ci.get("pr_auc")),
                with_ci(test["roc_auc"], ci.get("roc_auc")),
            ]
        )
    headers = [
        "Variant",
        "Model",
        "Test rows",
        "Test fake share",
        "PR-AUC [95% CI]",
        "ROC-AUC [95% CI]",
    ]
    return markdown_table(headers, rows)


def threshold_table(metrics: dict[str, Any], name: str) -> str:
    """Test metrics at one named threshold (0.5, t_f1 or t_r90) for every model."""
    rows = []
    for variant, model, result, _ in model_results(metrics):
        m = result["test"]["at_threshold"].get(name)
        if m is None:
            continue
        ci = result.get("test_bootstrap_ci", {}).get("at_threshold", {}).get(name, {})
        rows.append(
            [
                variant,
                model,
                fmt(m["threshold"]),
                with_ci(m["precision"], ci.get("precision")),
                with_ci(m["recall"], ci.get("recall")),
                with_ci(m["f1"], ci.get("f1")),
                fmt(m["macro_f1"]),
                fmt(m["accuracy"]),
                *(fmt(m[k]) for k in ("tn", "fp", "fn", "tp")),
            ]
        )
    headers = [
        "Variant",
        "Model",
        "Threshold",
        "Precision",
        "Recall",
        "F1 (fake)",
        "Macro-F1",
        "Accuracy",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    return markdown_table(headers, rows)


def mcnemar_table(metrics: dict[str, Any]) -> str:
    """Exact McNemar results (b = only A right, c = only B right)."""
    rows = [
        [t["model_a"], t["model_b"], t["threshold"], fmt(t["b"]), fmt(t["c"]), fmt(t["p_value"], 4)]
        for t in metrics.get("mcnemar", [])
    ]
    return markdown_table(["Model A", "Model B", "Threshold", "b", "c", "p-value"], rows)


def _trio(pair: dict[str, Any]) -> list[str]:
    """Reported, realistic and difference as three cells."""
    return [fmt(pair["reported"]), fmt(pair["realistic"]), fmt(pair["difference"])]


def reported_vs_realistic_tables(metrics: dict[str, Any]) -> str:
    """E1: reported (balanced before split) vs realistic (E1c) performance."""
    comparisons = metrics.get("reported_vs_realistic", [])
    head = ["Reported", "Realistic"]
    score_rows, recall_rows = [], []
    for c in comparisons:
        names = [c["reported"], c["realistic"]]
        score_rows.append(
            names + _trio(c["test_fake_share"]) + _trio(c["pr_auc"]) + _trio(c["roc_auc"])
        )
        for t_name, pair in c["recall"].items():
            recall_rows.append(names + [t_name] + _trio(pair))
    trio = ["reported", "realistic", "difference"]
    score_headers = (
        head
        + [f"Test fake share {x}" for x in trio]
        + [f"PR-AUC {x}" for x in trio]
        + [f"ROC-AUC {x}" for x in trio]
    )
    recall_headers = head + ["Threshold"] + [f"Recall (fake) {x}" for x in trio]
    return (
        markdown_table(score_headers, score_rows)
        + "\n\n"
        + markdown_table(recall_headers, recall_rows)
    )


def task_markdown(experiment_id: str, run: RunData) -> str:
    """Tables for a task experiment (E5-E8), headed by its source run."""
    from cvx.report.task_tables import TASK_TABLES

    git = run.env.get("git", {})
    parts = [
        f"# {experiment_id}",
        f"Source: `{run.path.parent.name}/{run.path.name}` · git `{git.get('commit', 'unknown')}`",
        *TASK_TABLES[run.metrics["task"]](run.metrics),
    ]
    return "\n\n".join(parts) + "\n"


def experiment_markdown(experiment_id: str, run: RunData) -> str:
    """All tables for one experiment, headed by the run folder they came from."""
    if run.metrics.get("task", "train") != "train":
        return task_markdown(experiment_id, run)
    metrics, git = run.metrics, run.env.get("git", {})
    parts = [
        f"# {experiment_id}",
        f"Source: `{run.path.parent.name}/{run.path.name}` · git `{git.get('commit', 'unknown')}`"
        f" · dataset sha256 `{run.env.get('dataset', {}).get('sha256', 'unknown')}`",
        "## Threshold-free test metrics",
        score_table(metrics),
    ]
    first = next(model_results(metrics), None)
    for name in first[2]["test"]["at_threshold"] if first else {}:
        parts += [f"## Test metrics at threshold `{name}`", threshold_table(metrics, name)]
    if metrics.get("mcnemar"):
        parts += ["## McNemar tests (exact)", mcnemar_table(metrics)]
    if metrics.get("reported_vs_realistic"):
        parts += ["## Reported vs realistic performance", reported_vs_realistic_tables(metrics)]
    return "\n\n".join(parts) + "\n"


def load_paper_values(path: str | Path) -> dict[str, Any] | None:
    """Read the owner's Career Verify values file (None if it does not exist)."""
    file = Path(path)
    if not file.is_file():
        return None
    return yaml.safe_load(file.read_text(encoding="utf-8")) or {}


def _paper_cell(paper: dict[str, Any] | None, key: str) -> str:
    """One Career Verify cell: the owner's value, 'not reported' (null), or a fill-in marker."""
    if paper is None or key not in paper or paper[key] == PAPER_FILL:
        return PAPER_TODO
    value = paper[key]
    return "not reported" if value is None else fmt(value)


def _run_cell(run: RunData | None, variant: str | None, key: str) -> str:
    """One cell from the LR model of ``variant`` in ``run`` at threshold 0.5."""
    if run is None:
        return MISSING
    variants = run.metrics["variants"]
    summary = variants.get(variant) if variant else next(iter(variants.values()))
    result = (summary or {}).get("models", {}).get("lr")
    if result is None:
        return MISSING
    if key in ("pr_auc", "roc_auc"):
        return fmt(result["test"]["threshold_free"][key])
    if key in ("test_rows", "test_fake_share"):
        return fmt(summary["split_summary"]["test"]["rows" if key == "test_rows" else "fake_share"])
    return fmt(result["test"]["at_threshold"][PAPER_THRESHOLD][key])


PAPER_ROWS = (
    ("Accuracy", "accuracy"),
    ("Precision (fake)", "precision"),
    ("Recall (fake)", "recall"),
    ("F1 (fake)", "f1"),
    ("PR-AUC", "pr_auc"),
    ("ROC-AUC", "roc_auc"),
    ("TN", "tn"),
    ("FP", "fp"),
    ("FN", "fn"),
    ("TP", "tp"),
    ("Test rows", "test_rows"),
    ("Test fake share", "test_fake_share"),
)


def paper_comparison_markdown(paper: dict[str, Any] | None, runs: dict[str, RunData]) -> str:
    """'Career Verify reported vs E0 reproduced vs E2 corrected' (LR, threshold 0.5)."""
    e0, e2 = runs.get("e0_reproduce"), runs.get("e2_protocol")
    rows = [
        [
            label,
            _paper_cell(paper, key),
            _run_cell(e0, None, key),
            _run_cell(e2, "e2a_p_cv", key),
            _run_cell(e2, "e2b_p_cvx", key),
        ]
        for label, key in PAPER_ROWS
    ]
    headers = [
        "Metric",
        "Career Verify (reported)",
        "E0 reproduced",
        "E2a corrected (P-CV)",
        "E2b corrected (P-CVX)",
    ]
    sources = ", ".join(f"`{r.path.parent.name}/{r.path.name}`" for r in (e0, e2) if r)
    source = (paper or {}).get("source", PAPER_TODO)
    if source == PAPER_FILL:
        source = PAPER_TODO
    intro = (
        f"Logistic Regression, threshold 0.5. Career Verify column: {source}. "
        f"Runs: {sources or MISSING}."
    )
    title = "# Career Verify reported vs E0 reproduced vs E2 corrected"
    return f"{title}\n\n{intro}\n\n{markdown_table(headers, rows)}\n"


def write_tables(runs_root: str | Path, out_dir: str | Path, paper_path: str | Path) -> list[Path]:
    """Write one Markdown file per experiment plus the paper comparison. Output: paths."""
    out, runs = Path(out_dir), latest_runs(runs_root)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for experiment_id, run in runs.items():
        path = out / f"{experiment_id}.md"
        path.write_text(experiment_markdown(experiment_id, run), encoding="utf-8")
        written.append(path)
    path = out / "career_verify_vs_e0_vs_e2.md"
    path.write_text(
        paper_comparison_markdown(load_paper_values(paper_path), runs), encoding="utf-8"
    )
    written.append(path)
    logger.info("Wrote %d table files to %s", len(written), out)
    return written
