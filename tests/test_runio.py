"""Tests for cvx.runio: run folders, env.json status, files and logs."""

import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from cvx import runio
from cvx.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_create_run_dir_never_overwrites(tmp_path: Path) -> None:
    first = runio.create_run_dir(tmp_path, "e0", stamp="20260101-000000")
    second = runio.create_run_dir(tmp_path, "e0", stamp="20260101-000000")
    assert first.name == "20260101-000000" and second.name == "20260101-000000-2"


def test_json_and_csv_round_trip(tmp_path: Path) -> None:
    runio.write_json(tmp_path / "a.json", {"x": [1, 2]})
    assert runio.read_json(tmp_path / "a.json") == {"x": [1, 2]}
    runio.write_csv(tmp_path / "a.csv", [{"a": 1, "b": "z"}], ["a", "b"])
    assert (tmp_path / "a.csv").read_text(encoding="utf-8").splitlines() == ["a,b", "1,z"]


def test_env_status_goes_running_to_complete(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "e0_reproduce.yaml")
    data = tmp_path / "data.csv"
    data.write_bytes(b"abc")
    env = runio.start_env(tmp_path, config, data)
    assert runio.read_json(tmp_path / "env.json")["status"] == "running"
    assert env["dataset"]["sha256"].startswith("ba7816bf")
    assert "scikit-learn" in env["libraries"] and "commit" in env["git"]
    runio.finish_env(tmp_path, env)
    finished = runio.read_json(tmp_path / "env.json")
    assert finished["status"] == "complete" and "finished_at_utc" in finished


def test_write_config_and_save_model(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "e0_reproduce.yaml")
    assert "experiment_id: e0_reproduce" in runio.write_config(tmp_path, config).read_text("utf-8")
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    assert runio.save_model(tmp_path, "lr_main", model).name == "lr_main.joblib"


def test_log_file(tmp_path: Path) -> None:
    handler = runio.attach_log_file(tmp_path)
    logging.getLogger("cvx.test").info("hello log")
    runio.detach_log_file(handler)
    assert "hello log" in (tmp_path / "run.log").read_text(encoding="utf-8")
