"""PR curves, ROC curves and confusion matrices as PNG files.

Uses matplotlib's object-oriented ``Figure`` (no pyplot), so it works without a
screen, in Colab and in tests. Used by the run folder and by scripts/make_figures.py.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

Curves = dict[str, tuple[np.ndarray, np.ndarray]]  # label -> (y_true, scores)


def save_figure(fig: Figure, path: str | Path) -> Path:
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
    return save_figure(fig, path)


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
    return save_figure(fig, path)


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
    return save_figure(fig, path)


def shap_bar_figure(items: list[tuple[str, float]], title: str) -> Figure:
    """Horizontal bar chart of (label, SHAP value); red = towards fake, green = genuine.

    Returned (not saved) so the app can show it with ``st.pyplot``. Bars also carry a
    +/- value label, so colour is not the only signal.
    """
    labels = [label for label, _ in items][::-1]
    values = [value for _, value in items][::-1]
    fig = Figure(figsize=(6.4, 0.42 * max(len(items), 1) + 1.15), facecolor="#ffffff")
    ax = fig.add_subplot(facecolor="#ffffff")
    colours = ["#dc2626" if v > 0 else "#16a34a" for v in values]
    ax.barh(range(len(values)), values, color=colours, height=0.64, edgecolor="none", zorder=3)
    ax.set_yticks(range(len(values)), labels, fontsize=9, color="#1e293b")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")
    ax.tick_params(axis="x", colors="#475569", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    max_abs = max((abs(v) for v in values), default=0.1) or 0.1
    ax.set_xlim(-max_abs * 1.32, max_abs * 1.32)
    for position, value in enumerate(values):
        ax.annotate(
            f"{value:+.3f}",
            (value, position),
            va="center",
            fontsize=8.5,
            fontweight="semibold",
            color="#991b1b" if value > 0 else "#166534",
            xytext=(5 if value >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
        )
    ax.axvline(0, color="#64748b", linewidth=1.0, zorder=4)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=10, loc="left")
    ax.set_xlabel(
        "SHAP value (log-odds):  + pushes towards fake  |  − pushes towards genuine",
        fontsize=8.5,
        color="#475569",
        labelpad=6,
    )
    fig.tight_layout()
    return fig
