"""Tests for cvx.data.profile. Expected values are counted by hand from conftest.py."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cvx.data.load import file_sha256
from cvx.data.profile import (
    build_profile,
    class_counts,
    main,
    metadata_fraud_rates,
    missing_rates,
    text_length_stats,
    top_values_among_fake,
    vocabulary_size,
)


def test_class_counts(tiny_postings: pd.DataFrame) -> None:
    assert class_counts(tiny_postings) == {"rows": 6, "genuine": 3, "fake": 3, "fraud_rate": 0.5}


def test_missing_rates_by_class(tiny_postings: pd.DataFrame) -> None:
    benefits = missing_rates(tiny_postings)["benefits"]
    # benefits is missing in rows 0, 1 (genuine) and 5 (fake).
    assert benefits["all"] == pytest.approx(3 / 6)
    assert benefits["fake"] == pytest.approx(1 / 3)
    assert benefits["genuine"] == pytest.approx(2 / 3)
    assert "job_id" not in missing_rates(tiny_postings)


def test_metadata_fraud_rates(tiny_postings: pd.DataFrame) -> None:
    rates = metadata_fraud_rates(tiny_postings)
    # Logo in rows 0, 1, 4 (all genuine); no logo in rows 2, 3, 5 (all fake).
    assert rates["has_company_logo"]["yes"] == {"rows": 3, "fake": 0, "fraud_rate": 0.0}
    assert rates["has_company_logo"]["no"] == {"rows": 3, "fake": 3, "fraud_rate": 1.0}
    # Salary given in rows 0, 1 (genuine) and 5 (fake).
    assert rates["salary_given"]["yes"]["rows"] == 3
    assert rates["salary_given"]["yes"]["fraud_rate"] == pytest.approx(1 / 3)
    assert set(rates) == {"telecommuting", "has_company_logo", "has_questions", "salary_given"}


def test_text_length_stats(make_postings) -> None:
    df = make_postings(
        [
            {"title": "ab", "description": "cd", "fraudulent": 1},  # "ab cd"  -> 5 chars
            {"title": "abc", "fraudulent": 1},  # "abc"    -> 3 chars
            {"title": "a", "benefits": "bcdefgh", "fraudulent": 0},  # "a bcdefgh" -> 9 chars
        ]
    )
    stats = text_length_stats(df)
    assert stats["fake"]["min"] == 3 and stats["fake"]["max"] == 5
    assert stats["fake"]["median"] == 4 and stats["fake"]["count"] == 2
    assert stats["genuine"]["mean"] == 9


def test_top_values_among_fake(tiny_postings: pd.DataFrame) -> None:
    assert top_values_among_fake(tiny_postings, "location") == [
        {"value": "US, TX, Houston", "fake_count": 2},
        {"value": "US, TX, Dallas", "fake_count": 1},
    ]
    assert top_values_among_fake(tiny_postings, "industry") == [
        {"value": "Financial Services", "fake_count": 1}
    ]


def test_vocabulary_size() -> None:
    assert vocabulary_size(pd.Series(["a b", "b c", ""])) == 3


def test_build_profile_duplicates(tiny_postings: pd.DataFrame) -> None:
    profile = build_profile(tiny_postings, near_dup_threshold=0.95)
    assert profile["counts_before_dedup"]["rows"] == 6
    assert profile["counts_after_exact_dedup"] == {
        "rows": 5,
        "genuine": 2,
        "fake": 3,
        "fraud_rate": 0.6,
    }
    dups = profile["duplicates"]
    assert dups["exact_duplicates_removed"] == 1
    # Rows 2 and 3 have identical text -> one near-duplicate group of two.
    near = dups["near_duplicates"]
    assert near["groups_with_more_than_one_row"] == 1
    assert near["rows_in_such_groups"] == 2
    assert near["groups_with_mixed_labels"] == 0
    assert set(profile["vocabulary_size_after_cleaning"]) == {"p_cv", "p_cvx"}


def test_main_writes_json(tmp_path: Path, tiny_postings: pd.DataFrame) -> None:
    csv_path = tmp_path / "postings.csv"
    tiny_postings.to_csv(csv_path, index=False)
    config = {
        "experiment_id": "profile_test",
        "seed": 42,
        "paths": {
            "data_csv": csv_path.as_posix(),
            "runs_dir": (tmp_path / "runs").as_posix(),
            "reports_dir": (tmp_path / "reports").as_posix(),
        },
        "preprocessing": "p_cvx",
        "split": {
            "method": "grouped_stratified",
            "ratios": [0.7, 0.15, 0.15],
            "near_dup_threshold": 0.95,
            "n_folds": 20,
        },
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": None,
            "sublinear_tf": True,
        },
    }
    config_path = tmp_path / "c.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config", str(config_path)]) == 0
    written = json.loads((tmp_path / "reports" / "data_profile.json").read_text("utf-8"))
    assert written["dataset"]["sha256"] == file_sha256(csv_path)
    assert written["counts_before_dedup"]["rows"] == 6
    for key in (
        "missing_rate_by_class",
        "metadata_fraud_rates",
        "text_length_chars",
        "top10_fake_locations",
        "top10_fake_industries",
        "duplicates",
    ):
        assert key in written
