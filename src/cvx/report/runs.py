"""Find and read run folders written by ``python -m cvx.run``."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cvx.runio import STATUS_COMPLETE, read_json

logger = logging.getLogger(__name__)

EXPERIMENTS = (
    "e0_reproduce",
    "e1_leakage",
    "e2_protocol",
    "e3_models",
    "e4_ablations",
    "e5_explain",
    "e6_faithfulness",
    "e7_audit",
    "e8_rules",
)


@dataclass(frozen=True)
class RunData:
    """One run folder's metrics, test predictions and environment record."""

    path: Path
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    env: dict[str, Any]


def latest_complete_run(runs_root: str | Path, experiment_id: str) -> Path | None:
    """Newest run folder of ``experiment_id`` whose env.json says 'complete' (or None).

    Folder names are UTC timestamps, so sorting by name sorts by time.
    """
    base = Path(runs_root) / experiment_id
    if not base.is_dir():
        return None
    for run_dir in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        env_file = run_dir / "env.json"
        if env_file.is_file() and read_json(env_file).get("status") == STATUS_COMPLETE:
            return run_dir
        logger.info("Skipping incomplete run %s", run_dir)
    return None


def load_run(run_dir: str | Path) -> RunData:
    """Read metrics.json, predictions.csv and env.json from a run folder."""
    path = Path(run_dir)
    # Task runs (E5-E8) have no predictions.csv.
    file = path / "predictions.csv"
    predictions = pd.read_csv(file, dtype={"row_id": str}) if file.is_file() else pd.DataFrame()
    return RunData(
        path, read_json(path / "metrics.json"), predictions, read_json(path / "env.json")
    )


def latest_runs(runs_root: str | Path) -> dict[str, RunData]:
    """The newest complete run of every known experiment that has one."""
    found = {}
    for experiment_id in EXPERIMENTS:
        run_dir = latest_complete_run(runs_root, experiment_id)
        if run_dir is None:
            logger.warning("No complete run for %s in %s", experiment_id, runs_root)
        else:
            found[experiment_id] = load_run(run_dir)
    return found
