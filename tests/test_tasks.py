"""End-to-end tests for E5-E8 (cvx.tasks) on the tiny exported model store."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from cvx import runio
from cvx.report.runs import load_run
from cvx.report.tables import experiment_markdown
from cvx.run import run_experiment

SPLIT = (
    "{method: grouped_stratified, ratios: [0.6, 0.2, 0.2], near_dup_threshold: 0.95, n_folds: 5}"
)
TFIDF = "{ngram_range: [1, 2], min_df: 1, max_df: 1.0, max_features: 500, sublinear_tf: true}"


def write_config(tmp_path: Path, tiny_runs: dict[str, Any], store: Path, body: str) -> Path:
    """A task config pointing at the tiny dataset and model store (same split as training)."""
    text = (
        "experiment_id: {name}\nseed: 42\n"
        f"paths: {{data_csv: {tiny_runs['data'].as_posix()}, runs_dir: {(tmp_path / 'runs').as_posix()},"  # noqa: E501
        " reports_dir: reports}\n"
        f"preprocessing: p_cvx\nsplit: {SPLIT}\ntfidf: {TFIDF}\n"
        "tuning: {method: grouped_cv, cv_folds: 3, scoring: average_precision}\n"
    ) + body.replace("STORE", store.as_posix())
    path = tmp_path / "task.yaml"
    path.write_text(text.replace("{name}", "task_test"), encoding="utf-8")
    return path


def test_e5_explain(tiny_runs, tiny_store, tmp_path) -> None:
    body = (
        "task: explain\nxai: {model_store: STORE, models: [lr_full, xgb_text], n_check: 10, "
        "n_examples: 2, n_global: 15, top_global: 5, lime_samples: 100, lime_features: 5}\n"
    )
    run_dir = run_experiment(write_config(tmp_path, tiny_runs, tiny_store, body))
    metrics = runio.read_json(run_dir / "metrics.json")
    assert metrics["all_checks_passed"] is True and metrics["task"] == "explain"
    assert metrics["models"]["lr_full"]["check"]["kind"] == "linear"
    assert len(metrics["models"]["xgb_text"]["global_top"]) <= 5
    lines = (run_dir / "explanations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and "reasons" in json.loads(lines[0])
    assert (run_dir / "figures" / "global_shap_lr_full.png").is_file()
    assert runio.read_json(run_dir / "env.json")["status"] == "complete"
    assert "SHAP correctness checks" in experiment_markdown("e5_explain", load_run(run_dir))


def test_e6_faithfulness(tiny_runs, tiny_store, tmp_path) -> None:
    body = (
        "task: faithfulness\nxai: {model_store: STORE, models: [lr_text], n_postings: 6, "
        "ks: [1, 3], random_draws: 2, lime_samples: 100, lime_features: 5, lime_seeds: 2}\n"
    )
    run_dir = run_experiment(write_config(tmp_path, tiny_runs, tiny_store, body))
    summary = runio.read_json(run_dir / "metrics.json")["models"]["lr_text"]
    assert 0 < summary["n_postings"] <= 6
    assert set(summary["means"]) == {"shap", "lime", "random"}
    assert len(summary["wilcoxon"]) == 3 and summary["wilcoxon"][0]["metric"] == "comp@1"
    with (run_dir / "per_posting.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["n_postings"] and "shap_comp@3" in rows[0]
    assert (run_dir / "figures" / "comprehensiveness_lr_text.png").is_file()
    text = experiment_markdown("e6_faithfulness", load_run(run_dir))
    assert "Wilcoxon" in text and "| shap |" in text


def test_e7_audit(tiny_runs, tiny_store, tmp_path) -> None:
    body = (
        "task: audit\nxai: {model_store: STORE, models: [lr_text], n_global: 15, "
        "top_global: 10, n_errors: 3, mask_tokens: [alpha]}\n"
    )
    run_dir = run_experiment(write_config(tmp_path, tiny_runs, tiny_store, body))
    result = runio.read_json(run_dir / "metrics.json")["models"]["lr_text"]
    assert "alpha" in result["flagged_tokens"]
    assert set(result["e7a_audit_columns"]["change"]) == {"pr_auc", "roc_auc", "f1@t_f1"}
    assert result["e7b_masked"]["masked_tokens"] == result["flagged_tokens"]
    with (run_dir / "errors.csv").open(encoding="utf-8") as handle:
        assert next(csv.reader(handle))[-1] == "cause"
    assert "E7a + audit columns" in experiment_markdown("e7_audit", load_run(run_dir))


def test_e8_rules(tiny_runs, tmp_path) -> None:
    run_dir = run_experiment(write_config(tmp_path, tiny_runs, tmp_path, "task: rules\n"))
    metrics = runio.read_json(run_dir / "metrics.json")
    # The test corpus's fake postings say "registration fee required" -> R1 fires only there.
    assert metrics["rules"]["R1"]["fake_hits"] > 0 and metrics["rules"]["R1"]["genuine_hits"] == 0
    assert metrics["not_validated"] == ["R3", "R4"]
    text = experiment_markdown("e8_rules", load_run(run_dir))
    assert "Rule hit rates" in text and "| R1 |" in text


def test_task_refuses_other_split(tiny_runs, tiny_store, tmp_path) -> None:
    body = "task: explain\nxai: {model_store: STORE, models: [lr_text]}\n"
    path = write_config(tmp_path, tiny_runs, tiny_store, body)
    path.write_text(
        path.read_text(encoding="utf-8").replace("n_folds: 5", "n_folds: 10"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split.n_folds"):
        run_experiment(path)
