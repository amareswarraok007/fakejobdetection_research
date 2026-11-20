"""Tests for cvx.report.figures: PR, ROC and confusion-matrix files from run folders."""

from pathlib import Path
from typing import Any

from cvx.report.figures import _safe, write_figures


def test_safe_names() -> None:
    assert _safe("0.5") == "0_5" and _safe("t_f1") == "t_f1"


def test_write_figures(tiny_runs: dict[str, Any], tmp_path: Path) -> None:
    written = write_figures(tiny_runs["root"], tmp_path)
    names = {p.name for p in written}
    assert "e3_models_main_pr.png" in names and "e3_models_main_roc.png" in names
    assert "e3_models_main_xgb_cm_t_r90.png" in names
    assert "e1_leakage_e1b_lr_cm_0_5.png" in names
    assert all(p.is_file() for p in written)
