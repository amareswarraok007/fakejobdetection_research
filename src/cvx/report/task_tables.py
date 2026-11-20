"""Markdown tables for the task experiments E5-E8, read only from their metrics.json."""

from typing import Any

from cvx.report.tables import fmt, markdown_table


def e5_tables(metrics: dict[str, Any]) -> list[str]:
    """Correctness checks and global top features per model."""
    models = metrics["models"]
    checks = [
        [
            m,
            r["check"]["kind"],
            r["check"]["check"],
            fmt(r["check"]["value"], 10),
            fmt(r["check"]["limit"], 6),
            str(r["check"]["passed"]),
        ]
        for m, r in models.items()
    ]
    parts = [
        "## SHAP correctness checks",
        markdown_table(["Model", "Kind", "Check", "Value", "Limit", "Passed"], checks),
    ]
    for model, result in models.items():
        rows = [
            [str(i + 1), f"`{r['feature']}`", fmt(r["mean_abs_shap"], 4)]
            for i, r in enumerate(result["global_top"])
        ]
        parts += [
            f"## Global top features: {model}",
            markdown_table(["Rank", "Feature", "Mean |SHAP|"], rows),
        ]
    return parts


def e6_tables(metrics: dict[str, Any]) -> list[str]:
    """Faithfulness by method, Wilcoxon tests, agreement, stability and runtime per model."""
    ks = metrics["settings"]["ks"]
    parts = []
    for model, s in metrics["models"].items():
        head = ["Method", "AOPC", *[f"comp@{k}" for k in ks], *[f"suff@{k}" for k in ks]]
        rows = [
            [
                m,
                fmt(v["aopc"]),
                *[fmt(v["comp"][str(k)]) for k in ks],
                *[fmt(v["suff"][str(k)]) for k in ks],
            ]
            for m, v in s["means"].items()
        ]
        tests = [
            [
                f"{t['a']} vs {t['b']}",
                t["metric"],
                fmt(t["n"]),
                fmt(t["median_difference"]),
                fmt(t["p_value"], 4),
            ]
            for t in s["wilcoxon"]
        ]
        other = [[name, fmt(value)] for name, value in s["agreement"].items()]
        other.append(["LIME stability (top-5 Jaccard)", fmt(s["lime_stability_top5"])])
        for method, r in s["seconds_per_explanation"].items():
            other.append([f"{method} seconds (mean / p95)", f"{fmt(r['mean'])} / {fmt(r['p95'])}"])
        parts += [
            f"## {model}: faithfulness ({s['n_postings']} postings)",
            markdown_table(head, rows),
            f"## {model}: Wilcoxon signed-rank tests",
            markdown_table(["Pair", "Metric", "n", "Median difference", "p-value"], tests),
            f"## {model}: agreement, stability, runtime",
            markdown_table(["Measure", "Value"], other),
        ]
    return parts


def _change_row(name: str, block: dict[str, Any]) -> list[str]:
    """One E7 row: metric changes versus the exported model."""
    change = block["change"]
    return [
        name,
        fmt(change["pr_auc"]),
        fmt(change["roc_auc"]),
        fmt(change["f1@t_f1"]),
        fmt(block.get("audit_share_top")),
    ]


def e7_tables(metrics: dict[str, Any]) -> list[str]:
    """E7a/E7b metric changes and flagged tokens per model."""
    parts = []
    for model, r in metrics["models"].items():
        rows = [
            _change_row("E7a + audit columns", r["e7a_audit_columns"]),
            _change_row("E7b masked tokens", r["e7b_masked"]),
        ]
        head = [
            "Re-train",
            "Δ PR-AUC",
            "Δ ROC-AUC",
            "Δ F1 @ t_f1",
            "Share of top features from audit columns",
        ]
        tokens = ", ".join(f"`{t}`" for t in r["flagged_tokens"]) or "none"
        parts += [f"## {model}", markdown_table(head, rows), f"Flagged and masked tokens: {tokens}"]
    return parts


def e8_tables(metrics: dict[str, Any]) -> list[str]:
    """Rule hit rates among fake and genuine postings."""

    def row(name: str, r: dict[str, Any]) -> list[str]:
        """Counts and rates for one rule or level."""
        return [
            name,
            fmt(r["fake_hits"]),
            fmt(r["fake_rate"]),
            fmt(r["genuine_hits"]),
            fmt(r["genuine_rate"]),
        ]

    head = ["Rule", "Fake hits", "Fake rate", "Genuine hits", "Genuine rate"]
    rows = [row(rule, r) for rule, r in metrics["rules"].items()] + [
        row("any rule", metrics["any_rule"])
    ]
    levels = [row(level, r) for level, r in metrics["rules_level"].items()]
    return [
        "## Rule hit rates",
        markdown_table(head, rows),
        "## Rules level",
        markdown_table(head, levels),
        f"Not validated: {', '.join(metrics['not_validated'])}. {metrics['note']}",
    ]


TASK_TABLES = {
    "explain": e5_tables,
    "faithfulness": e6_tables,
    "audit": e7_tables,
    "rules": e8_tables,
}
