"""Reproducibility check (PRD M8): do two runs of the same config give the same numbers?

Every number in the two ``metrics.json`` files is compared (same keys, values equal
within a tolerance). Run ids, timings and other bookkeeping fields are skipped.
"""

import math
from pathlib import Path
from typing import Any

from cvx.runio import read_json

SKIPPED_KEYS = {
    "run_id",
    "fit_seconds",
    "seconds",
    "lime_seconds",
    "shap_seconds",
    "seconds_per_explanation",
}


def differences(a: Any, b: Any, tolerance: float = 1e-9, path: str = "") -> list[str]:
    """Human-readable list of places where ``a`` and ``b`` differ (empty = identical)."""
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(set(a) | set(b), key=str):
            if key in SKIPPED_KEYS:
                continue
            if key not in a or key not in b:
                found.append(f"{path}/{key}: present in only one run")
                continue
            found += differences(a[key], b[key], tolerance, f"{path}/{key}")
        return found
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list lengths {len(a)} vs {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, tolerance, f"{path}[{i}]")
        ]
    numeric = (int, float)
    if isinstance(a, numeric) and isinstance(b, numeric) and not isinstance(a, bool):
        same = math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        return [] if same else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def compare_runs(run_a: str | Path, run_b: str | Path, tolerance: float = 1e-9) -> list[str]:
    """Differences between the ``metrics.json`` of two run folders."""
    return differences(
        read_json(Path(run_a) / "metrics.json"), read_json(Path(run_b) / "metrics.json"), tolerance
    )
