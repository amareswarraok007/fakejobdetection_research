"""Tests for cvx.report.runs and cvx.report.tables (tables read only from run folders)."""

from pathlib import Path
from typing import Any

from cvx import runio
from cvx.report.runs import latest_complete_run, latest_runs, load_run
from cvx.report.tables import (
    PAPER_TODO,
    experiment_markdown,
    fmt,
    markdown_table,
    paper_comparison_markdown,
    with_ci,
    write_tables,
)


def test_fmt_and_ci() -> None:
    assert fmt(None) == "n/a" and fmt(3) == "3" and fmt(0.12345) == "0.123"
    assert with_ci(0.5, {"low": 0.4, "high": 0.6}) == "0.500 [0.400, 0.600]"
    assert with_ci(0.5, None) == "0.500"


def test_markdown_table() -> None:
    assert markdown_table(["a", "b"], [["1", "2"]]) == "| a | b |\n| --- | --- |\n| 1 | 2 |"


def test_latest_complete_run_skips_incomplete(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    good = tiny_runs["runs"]["e0_reproduce"]
    assert latest_complete_run(tiny_runs["root"], "e0_reproduce") == good
    broken = tmp_path / "e0_reproduce" / "29991231-000000"
    broken.mkdir(parents=True)
    runio.write_json(broken / "env.json", {"status": "running"})
    assert latest_complete_run(tmp_path, "e0_reproduce") is None


def test_experiment_markdown_numbers_come_from_metrics(tiny_runs: dict[str, Any]) -> None:
    run = load_run(tiny_runs["runs"]["e3_models"])
    text = experiment_markdown("e3_models", run)
    pr_auc = run.metrics["variants"]["main"]["models"]["lr"]["test"]["threshold_free"]["pr_auc"]
    assert fmt(pr_auc) in text
    assert "## McNemar tests (exact)" in text and "`t_r90`" in text
    assert run.path.name in text  # every table names its source run


def test_e1_tables_have_reported_vs_realistic(tiny_runs: dict[str, Any]) -> None:
    text = experiment_markdown("e1_leakage", load_run(tiny_runs["runs"]["e1_leakage"]))
    assert "Reported vs realistic performance" in text
    assert "ROC-AUC reported" in text and "Recall (fake) realistic" in text


def test_paper_table_uses_owner_file(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    runs = latest_runs(tiny_runs["root"])
    assert PAPER_TODO in paper_comparison_markdown(None, runs)
    text = paper_comparison_markdown({"source": "Table 9", "accuracy": 0.25, "pr_auc": None}, runs)
    assert "Table 9" in text and "0.250" in text and "not reported" in text
    assert "n/a" in text  # no E2 run in the tiny set


def test_write_tables(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    written = write_tables(tiny_runs["root"], tmp_path, tmp_path / "missing.yaml")
    names = {p.name for p in written}
    assert {
        "e0_reproduce.md",
        "e1_leakage.md",
        "e3_models.md",
        "e4_ablations.md",
        "career_verify_vs_e0_vs_e2.md",
    } == names
