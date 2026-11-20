"""Report figures from run folders: PR curves, ROC curves and confusion matrices."""

import logging
import re
from pathlib import Path

from cvx.evaluate.plots import plot_confusion, plot_pr_curves, plot_roc_curves
from cvx.report.runs import RunData, latest_runs

logger = logging.getLogger(__name__)


def _safe(name: str) -> str:
    """File-name-safe version of a label ('0.5' -> '0_5')."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def experiment_figures(experiment_id: str, run: RunData, out_dir: Path) -> list[Path]:
    """PR and ROC curves per variant (all models together) and one confusion matrix
    per variant, model and threshold, from ``predictions.csv`` and ``metrics.json``."""
    written: list[Path] = []
    if run.predictions.empty:  # task runs (E5-E8) save their own figures
        return written
    for variant, rows in run.predictions.groupby("variant", sort=False):
        curves = {
            model: (part["label"].to_numpy(), part["score"].to_numpy())
            for model, part in rows.groupby("model", sort=False)
        }
        stem = f"{experiment_id}_{_safe(variant)}"
        written.append(
            plot_pr_curves(curves, out_dir / f"{stem}_pr.png", f"{experiment_id}: {variant}")
        )
        written.append(
            plot_roc_curves(curves, out_dir / f"{stem}_roc.png", f"{experiment_id}: {variant}")
        )
        for model, result in run.metrics["variants"][variant]["models"].items():
            for t_name, m in result["test"]["at_threshold"].items():
                path = out_dir / f"{stem}_{model}_cm_{_safe(t_name)}.png"
                title = f"{variant}/{model} @ {t_name} ({m['threshold']:.3f})"
                written.append(plot_confusion(m, path, title))
    return written


def write_figures(runs_root: str | Path, out_dir: str | Path) -> list[Path]:
    """Figures for the newest complete run of every experiment. Output: file paths."""
    out = Path(out_dir)
    written = []
    for experiment_id, run in latest_runs(runs_root).items():
        written += experiment_figures(experiment_id, run, out)
    logger.info("Wrote %d figures to %s", len(written), out)
    return written
