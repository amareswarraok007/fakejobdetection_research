"""PR curves, ROC curves and confusion matrices as PNG files.

Uses matplotlib's object-oriented ``Figure`` (no pyplot), so it works without a
screen, in Colab and in tests. Used by the run folder and by scripts/make_figures.py.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

Curves = dict[str, tuple[np.ndarray, np.ndarray]]  # label -> (y_true, scores)


def _save(fig: Figure, path: str | Path) -> Path:
    """Save a figure as PNG, creating the folder if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out


def plot_pr_curves(curves: Curves, path: str | Path, title: str) -> Path:
    """Precision-recall curve for each (label -> labels, scores) pair on one chart."""
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot()
    for label, (y, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, label=label)
    ax.set(
        xlabel="Recall (fake)", ylabel="Precision (fake)", title=title, xlim=(0, 1), ylim=(0, 1.02)
    )
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_roc_curves(curves: Curves, path: str | Path, title: str) -> Path:
    """ROC curve for each (label -> labels, scores) pair, with the chance line."""
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot()
    for label, (y, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(fontsize="small")
    return _save(fig, path)


def plot_confusion(counts: dict[str, int], path: str | Path, title: str) -> Path:
    """2x2 confusion matrix with the count written in every cell."""
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig = Figure(figsize=(3.6, 3.2))
    ax = fig.add_subplot()
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        colour = "white" if value > matrix.max() / 2 else "black"
        ax.text(col, row, str(value), ha="center", va="center", color=colour)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["genuine", "fake"],
        yticklabels=["genuine", "fake"],
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    return _save(fig, path)
