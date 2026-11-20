"""End-to-end tests for cvx.run: a complete run folder (PRD Section 10.4) for tiny runs."""

import csv
from pathlib import Path
from typing import Any

import pytest

from cvx import runio
from cvx.run import main

RUN_FILES = (
    "config.yaml",
    "env.json",
    "metrics.json",
    "cv_results.csv",
    "predictions.csv",
    "run.log",
)


@pytest.mark.parametrize("name", ["e0_reproduce", "e1_leakage", "e3_models"])
def test_run_folder_is_complete(tiny_runs: dict[str, Any], name: str) -> None:
    run_dir: Path = tiny_runs["runs"][name]
    for file_name in RUN_FILES:
        assert (run_dir / file_name).is_file(), file_name
    assert list((run_dir / "figures").glob("pr_*.png"))
    assert runio.read_json(run_dir / "env.json")["status"] == "complete"


def test_predictions_cover_test_split_only(tiny_runs: dict[str, Any]) -> None:
    run_dir = tiny_runs["runs"]["e0_reproduce"]
    metrics = runio.read_json(run_dir / "metrics.json")
    test_rows = metrics["variants"]["main"]["split_summary"]["test"]["rows"]
    with (run_dir / "predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == test_rows
    assert set(rows[0]) == {"variant", "model", "row_id", "label", "score"}


def test_e0_metrics_structure(tiny_runs: dict[str, Any]) -> None:
    metrics = runio.read_json(tiny_runs["runs"]["e0_reproduce"] / "metrics.json")
    lr = metrics["variants"]["main"]["models"]["lr"]
    assert set(lr["test"]["at_threshold"]) == {"0.5", "t_f1", "t_r90"}
    assert lr["best_params"]["C"] in (0.1, 1.0, 10.0)
    assert lr["test_bootstrap_ci"]["n_resamples"] == 20
    assert set(lr["chosen_thresholds"]) == {"t_f1", "t_r90"}


def test_e1_reported_vs_realistic(tiny_runs: dict[str, Any]) -> None:
    metrics = runio.read_json(tiny_runs["runs"]["e1_leakage"] / "metrics.json")
    variants = metrics["variants"]
    assert variants["e1a"]["split_summary"]["test"]["fake_share"] == pytest.approx(0.5, abs=0.1)
    assert variants["e1b"]["prevectorized"] is True
    # E1c keeps the natural prevalence (30 of 90 postings are fake in the test corpus).
    assert variants["e1c"]["split_summary"]["test"]["fake_share"] == pytest.approx(1 / 3)
    comparisons = metrics["reported_vs_realistic"]
    assert [c["reported"] for c in comparisons] == ["e1a/lr", "e1b/lr"]
    first = comparisons[0]
    assert set(first) >= {"pr_auc", "roc_auc", "recall", "test_fake_share"}
    pair = first["pr_auc"]
    assert pair["difference"] == pytest.approx(pair["reported"] - pair["realistic"])


def test_e3_mcnemar_export_and_cv(tiny_runs: dict[str, Any]) -> None:
    run_dir = tiny_runs["runs"]["e3_models"]
    metrics = runio.read_json(run_dir / "metrics.json")
    assert set(metrics["variants"]["main"]["models"]) == {"nb", "linear_svm", "rf", "lr", "xgb"}
    tests = metrics["mcnemar"]
    assert {t["threshold"] for t in tests} == {"0.5", "t_f1", "t_r90"}
    assert all(t["model_a"] == "main/lr" and t["model_b"] == "main/xgb" for t in tests)
    assert all(isinstance(t["b"], int) and isinstance(t["c"], int) for t in tests)
    assert len(list((run_dir / "models").glob("*.joblib"))) == 5
    with (run_dir / "cv_results.csv").open(encoding="utf-8") as handle:
        models_tuned = {row["model"] for row in csv.DictReader(handle)}
    assert "rf" not in models_tuned and "xgb" in models_tuned  # rf has one setting


def test_main_cli(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    config = tiny_runs["runs"]["e0_reproduce"] / "config.yaml"
    assert (
        main(
            ["--config", str(config), "--data", str(tiny_runs["data"]), "--runs-dir", str(tmp_path)]
        )
        == 0
    )
    assert (tmp_path / "e0_reproduce").is_dir()


def test_config_without_models_is_rejected(tmp_path: Path) -> None:
    from cvx.run import run_experiment

    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="no models"):
        run_experiment(repo / "configs" / "base.yaml", runs_dir=tmp_path)
