"""Data profile: every statistic listed in PRD Section 6.3, computed from the CSV.

Nothing here is typed by hand: every number in ``reports/data_profile.json`` is
computed by the functions below from the dataset file.

Scope of each statistic:
- Row and class counts: before AND after exact-duplicate removal.
- Near-duplicate groups: on the data after exact-duplicate removal (what CVX uses).
- Everything else (missing values, metadata fraud rates, text length, top
  locations/industries, vocabulary size): on the full dataset as downloaded.

Run from the repository root:
    python -m cvx.data.profile --config configs/base.yaml [--out reports/data_profile.json]
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cvx.config import load_config
from cvx.data import clean, dedup
from cvx.data.load import (
    BINARY_FIELDS,
    ID_COLUMN,
    LABEL_COLUMN,
    SALARY_FIELD,
    TEXT_FIELDS,
    file_sha256,
    is_missing,
    load_postings,
)

logger = logging.getLogger(__name__)


def class_counts(df: pd.DataFrame) -> dict[str, Any]:
    """Rows, genuine count, fake count and fraud rate (share of rows that are fake)."""
    fake = int((df[LABEL_COLUMN] == 1).sum())
    rows = int(len(df))
    return {
        "rows": rows,
        "genuine": rows - fake,
        "fake": fake,
        "fraud_rate": fake / rows if rows else None,
    }


def missing_rates(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Share of missing (NaN or blank) values per column: overall, fake and genuine."""
    is_fake = df[LABEL_COLUMN] == 1
    result: dict[str, dict[str, float | None]] = {}
    for column in df.columns:
        if column in (ID_COLUMN, LABEL_COLUMN):
            continue
        missing = is_missing(df[column])
        result[column] = {
            "all": _mean(missing),
            "fake": _mean(missing[is_fake]),
            "genuine": _mean(missing[~is_fake]),
        }
    return result


def metadata_fraud_rates(df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    """For logo, questions, remote and salary-given: rows, fakes and fraud rate per yes/no."""
    flags = {name: df[name] == 1 for name in BINARY_FIELDS}
    flags["salary_given"] = ~is_missing(df[SALARY_FIELD])
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for name, flag in flags.items():
        result[name] = {}
        for value_name, rows in (("yes", flag), ("no", ~flag)):
            subset = df.loc[rows]
            stats = class_counts(subset)
            result[name][value_name] = {
                "rows": stats["rows"],
                "fake": stats["fake"],
                "fraud_rate": stats["fraud_rate"],
            }
    return result


def raw_joined_text(df: pd.DataFrame) -> pd.Series:
    """The five raw text fields joined with spaces (empty fields skipped), before cleaning."""
    parts = df[list(TEXT_FIELDS)].astype("string").fillna("")
    return parts.agg(lambda row: " ".join(v for v in row if v.strip()), axis=1)


def text_length_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Distribution of raw joined-text length in characters, separately for fake and genuine."""
    lengths = raw_joined_text(df).str.len()
    result: dict[str, dict[str, float]] = {}
    for name, rows in (("fake", df[LABEL_COLUMN] == 1), ("genuine", df[LABEL_COLUMN] == 0)):
        values = lengths[rows]
        if values.empty:
            continue
        quantiles = values.quantile([0.25, 0.5, 0.75])
        result[name] = {
            "count": int(values.size),
            "min": float(values.min()),
            "q25": float(quantiles[0.25]),
            "median": float(quantiles[0.5]),
            "q75": float(quantiles[0.75]),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return result


def top_values_among_fake(df: pd.DataFrame, column: str, n: int = 10) -> list[dict[str, Any]]:
    """The ``n`` most common non-missing values of ``column`` among fake postings."""
    fake = df.loc[(df[LABEL_COLUMN] == 1) & ~is_missing(df[column]), column]
    counts = fake.astype(str).str.strip().value_counts().head(n)
    return [{"value": value, "fake_count": int(count)} for value, count in counts.items()]


def vocabulary_size(texts: pd.Series) -> int:
    """Number of distinct space-separated words across all cleaned texts."""
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(text.split())
    return len(vocabulary)


def duplicate_profile(df: pd.DataFrame, near_dup_threshold: float) -> dict[str, Any]:
    """Exact-duplicate count and near-duplicate group summary (after exact removal)."""
    deduped, n_exact = dedup.drop_exact_duplicates(df)
    texts = clean.p_cvx(deduped, with_labels=False)
    groups = dedup.near_duplicate_groups(texts.tolist(), near_dup_threshold)
    return {
        "exact_duplicates_removed": n_exact,
        "near_duplicates": {
            "threshold": near_dup_threshold,
            "text": "p_cvx without field labels, TfidfVectorizer defaults",
            **dedup.group_summary(groups, deduped[LABEL_COLUMN].to_numpy()),
        },
        "counts_after_exact_removal": class_counts(deduped),
    }


def build_profile(df: pd.DataFrame, near_dup_threshold: float) -> dict[str, Any]:
    """Compute every PRD Section 6.3 statistic. Output: a JSON-ready dictionary."""
    duplicates = duplicate_profile(df, near_dup_threshold)
    return {
        "counts_before_dedup": class_counts(df),
        "counts_after_exact_dedup": duplicates.pop("counts_after_exact_removal"),
        "duplicates": duplicates,
        "missing_rate_by_class": missing_rates(df),
        "metadata_fraud_rates": metadata_fraud_rates(df),
        "text_length_chars": text_length_stats(df),
        "top10_fake_locations": top_values_among_fake(df, "location"),
        "top10_fake_industries": top_values_among_fake(df, "industry"),
        "vocabulary_size_after_cleaning": {
            "p_cv": vocabulary_size(clean.p_cv(df)),
            "p_cvx": vocabulary_size(clean.p_cvx(df, with_labels=False)),
        },
        "scope_note": (
            "Duplicates: near-duplicate groups are computed after exact-duplicate removal. "
            "All other statistics use the full dataset as downloaded."
        ),
    }


def write_profile(profile: dict[str, Any], out_path: str | Path) -> Path:
    """Write the profile as indented JSON (creating folders as needed). Output: the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: load config and CSV, build the profile, write JSON."""
    parser = argparse.ArgumentParser(description="Write reports/data_profile.json")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/base.yaml")
    parser.add_argument("--out", default=None, help="Output JSON (default: <reports_dir>/...)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    csv_path = Path(config.paths.data_csv)
    df = load_postings(csv_path)
    threshold = config.split.near_dup_threshold
    if threshold is None:
        raise ValueError("The config must set split.near_dup_threshold to profile duplicates.")
    profile = {
        "dataset": {"file": csv_path.name, "sha256": file_sha256(csv_path)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **build_profile(df, threshold),
    }
    out = args.out or Path(config.paths.reports_dir) / "data_profile.json"
    logger.info("Wrote %s", write_profile(profile, out))
    return 0


def _mean(mask: pd.Series) -> float | None:
    """Mean of a boolean Series as a float, or None when the Series is empty."""
    return float(mask.mean()) if len(mask) else None


if __name__ == "__main__":
    raise SystemExit(main())
