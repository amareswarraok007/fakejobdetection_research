"""Tests for cvx.report.compare (M8 reproducibility check)."""

from pathlib import Path
from typing import Any

from cvx.report.compare import compare_runs, differences
from cvx.run import run_experiment


def test_differences_hand_examples() -> None:
    assert differences({"a": 1.0, "run_id": "x"}, {"a": 1.0 + 1e-12, "run_id": "y"}) == []
    assert differences({"a": [1, 2]}, {"a": [1, 3]}) == ["/a[1]: 2 vs 3"]
    assert differences({"a": 1}, {"b": 1}) == [
        "/a: present in only one run",
        "/b: present in only one run",
    ]


def test_rerun_of_e0_gives_identical_metrics(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    stored = tiny_runs["runs"]["e0_reproduce"]
    fresh = run_experiment(stored / "config.yaml", tiny_runs["data"], tmp_path)
    assert compare_runs(stored, fresh) == []
