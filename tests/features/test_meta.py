"""Tests for cvx.features.meta: every metadata feature on a hand-checked posting."""

import math

import numpy as np
import pytest

from cvx.features.meta import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    count_money,
    metadata_frame,
    uppercase_ratio,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Earn $500 per week or ₹2000 per day", 4),  # $, per week, ₹, per day
        ("Rs. 5000 and INR 300", 2),
        ("Working hours. Inroads", 0),  # 'rs.' and 'inr' only as whole words
        ("", 0),
    ],
)
def test_count_money(text: str, expected: int) -> None:
    assert count_money(text) == expected


def test_uppercase_ratio() -> None:
    assert uppercase_ratio("ABcd") == 0.5
    assert uppercase_ratio("123 !!") == 0.0


def test_metadata_frame_hand_values(make_postings) -> None:
    df = make_postings(
        [
            {
                "title": "URGENT Job",
                "description": "<b>Earn $500</b> per week!! Mail #EMAIL_ab12# or #URL_cd34#",
                "benefits": "  ",
                "salary_range": "100-200",
                "has_company_logo": 1,
                "employment_type": "Full-time",
            }
        ]
    )
    meta = metadata_frame(df).iloc[0]
    # Cleaned text without labels: "urgent job earn 500 per week mail emailtoken or urltoken"
    unlabelled = "urgent job earn 500 per week mail emailtoken or urltoken"
    assert meta["log_text_length"] == pytest.approx(math.log1p(len(unlabelled)))
    assert (meta["n_url"], meta["n_email"], meta["n_phone"]) == (1, 1, 0)
    assert meta["money_mentions"] == 2  # "$" and "per week"
    assert meta["exclamation_count"] == 2
    # Masked raw text keeps case but the placeholders become lowercase tokens:
    # upper letters U,R,G,E,N,T,J,E,M = 9.
    letters = "URGENTJobEarnperweekMailemailtokenorurltoken"
    assert meta["uppercase_ratio"] == pytest.approx(9 / len(letters))
    assert meta["salary_given"] == 1 and meta["has_company_logo"] == 1
    assert meta["company_profile_empty"] == 1 and meta["benefits_empty"] == 1
    assert meta["requirements_empty"] == 1
    assert meta["employment_type"] == "Full-time" and meta["function"] == "Unknown"


def test_metadata_frame_columns_and_index(tiny_postings) -> None:
    subset = tiny_postings.iloc[[4, 2]]
    meta = metadata_frame(subset)
    assert list(meta.columns) == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert list(meta.index) == [4, 2]
    assert np.isfinite(meta[list(NUMERIC_FEATURES)].to_numpy()).all()
