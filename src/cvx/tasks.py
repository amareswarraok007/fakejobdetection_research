"""Run the non-training experiments (config ``task``): E5 explain, E6 faithfulness,
E7 audit (all on the exported models), and E8 rules (on the data only)."""

from pathlib import Path
from typing import Any

import pandas as pd

from cvx.audit.experiment import run_e7
from cvx.config import Config
from cvx.explain.context import make_context
from cvx.explain.experiment import run_e5
from cvx.rules.experiment import run_e8
from cvx.xai_eval.experiment import run_e6

_MODEL_TASKS = {"explain": run_e5, "faithfulness": run_e6, "audit": run_e7}


def run_task(
    config: Config, postings: pd.DataFrame, data_path: Path, run_dir: Path
) -> dict[str, Any]:
    """Run the config's task and return its metrics (files are written into ``run_dir``)."""
    if config.task == "rules":
        return run_e8(postings)
    if config.task not in _MODEL_TASKS:
        raise ValueError(f"No runner for task {config.task!r}.")
    context = make_context(config, postings, data_path)
    return _MODEL_TASKS[config.task](context, run_dir)
