"""Run folders (PRD Section 10.4): create them and write their files.

Layout: ``<runs_dir>/<experiment_id>/<UTC timestamp>/`` containing ``config.yaml``,
``env.json``, ``metrics.json``, ``cv_results.csv``, ``predictions.csv``, ``figures/``,
``run.log`` and, if the config asks, ``models/``.

``env.json`` starts with ``"status": "running"`` and becomes ``"complete"`` only after
every other file is written, so an interrupted Colab run is easy to recognise.
"""

import csv
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import yaml

from cvx.config import Config, config_to_dict
from cvx.data.load import file_sha256

LIBRARIES = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",
    "nltk",
    "shap",
    "lime",
    "matplotlib",
    "PyYAML",
    "joblib",
    "threadpoolctl",
)
STATUS_RUNNING, STATUS_COMPLETE = "running", "complete"


def utc_now() -> str:
    """Current UTC time as an ISO string (seconds)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_run_dir(runs_root: str | Path, experiment_id: str, stamp: str | None = None) -> Path:
    """Create ``runs_root/experiment_id/<timestamp>/`` (adds -2, -3 … if it already exists)."""
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = Path(runs_root) / experiment_id
    candidate, n = base / stamp, 2
    while candidate.exists():
        candidate, n = base / f"{stamp}-{n}", n + 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as indented UTF-8 JSON."""
    out = Path(path)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> Path:
    """Write ``rows`` (dicts) to CSV with the given column order."""
    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_config(run_dir: Path, config: Config) -> Path:
    """Save the merged config exactly as it was run."""
    text = yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=True)
    path = run_dir / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def library_versions() -> dict[str, str | None]:
    """Installed version of each library that can affect results (None if missing)."""
    versions: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def git_info(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Current git commit and whether there are uncommitted changes ('unknown' if no git)."""

    def git(*args: str) -> str:
        """Run one git command and return its output."""
        result = subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=20, check=True
        )
        return result.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def start_env(run_dir: Path, config: Config, data_path: Path) -> dict[str, Any]:
    """Write ``env.json`` with status 'running' and return its content."""
    env = {
        "status": STATUS_RUNNING,
        "experiment_id": config.experiment_id,
        "started_at_utc": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        "git": git_info(),
        "dataset": {"file": data_path.name, "sha256": file_sha256(data_path)},
        "seed": config.seed,
    }
    write_json(run_dir / "env.json", env)
    return env


def finish_env(run_dir: Path, env: dict[str, Any]) -> None:
    """Mark the run complete (call this last)."""
    write_json(
        run_dir / "env.json", {**env, "status": STATUS_COMPLETE, "finished_at_utc": utc_now()}
    )


def attach_log_file(run_dir: Path) -> logging.Handler:
    """Copy log messages from the ``cvx`` package into ``run.log`` (never posting text)."""
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    package_logger = logging.getLogger("cvx")
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    return handler


def detach_log_file(handler: logging.Handler) -> None:
    """Stop writing to ``run.log`` and close the file."""
    logging.getLogger("cvx").removeHandler(handler)
    handler.close()


def save_model(run_dir: Path, name: str, estimator: Any) -> Path:
    """Save a fitted pipeline to ``models/<name>.joblib`` (compressed)."""
    path = run_dir / "models" / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path, compress=3)
    return path
