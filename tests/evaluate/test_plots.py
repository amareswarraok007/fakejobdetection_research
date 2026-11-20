"""Tests for cvx.evaluate.plots: each function writes a PNG file."""

from pathlib import Path

import numpy as np

from cvx.evaluate.plots import plot_confusion, plot_pr_curves, plot_roc_curves

Y = np.array([1, 1, 0, 0, 0])
S = np.array([0.9, 0.4, 0.6, 0.2, 0.1])


def test_curves_written(tmp_path: Path) -> None:
    curves = {"lr": (Y, S), "xgb": (Y, S[::-1])}
    for plot, name in ((plot_pr_curves, "pr.png"), (plot_roc_curves, "sub/roc.png")):
        path = plot(curves, tmp_path / name, "title")
        assert path.is_file() and path.read_bytes()[:4] == b"\x89PNG"


def test_confusion_written(tmp_path: Path) -> None:
    path = plot_confusion({"tn": 2, "fp": 1, "fn": 1, "tp": 1}, tmp_path / "cm.png", "cm")
    assert path.is_file()
