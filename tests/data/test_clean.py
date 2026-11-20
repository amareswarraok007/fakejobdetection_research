"""Tests for cvx.data.clean: P-CV and P-CVX on hand-written inputs with hand-derived outputs."""

import numpy as np
import pandas as pd
import pytest

from cvx.data.clean import (
    UNKNOWN,
    build_text,
    clean_text_p_cv,
    clean_text_p_cvx,
    fill_missing_categories,
    p_cv,
    p_cvx,
)

# ---------------------------------------------------------------- P-CV


def test_p_cv_lowercase_stopwords_numbers_punctuation_lemmas() -> None:
    # "the", "are" are stop words; "3" and "!" are removed; "managers" -> "manager".
    assert clean_text_p_cv("The 3 Managers are RUNNING fast!") == "manager running fast"


def test_p_cv_removes_no_and_not() -> None:
    # NLTK's English stop-word list contains "no" and "not", so P-CV drops them.
    assert clean_text_p_cv("No experience, not required") == "experience required"


def test_p_cv_missing_value_is_empty() -> None:
    assert clean_text_p_cv(np.nan) == ""
    assert clean_text_p_cv(None) == ""


def test_p_cv_joins_five_fields_without_labels(make_postings) -> None:
    df = make_postings([{"title": "Sales Managers", "description": "Selling shoes"}])
    assert p_cv(df).tolist() == ["sale manager selling shoe"]


# ---------------------------------------------------------------- P-CVX


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>Sell shoes &amp; boots.</p>", "sell shoes boots"),
        ("Call #PHONE_abc123# or mail#EMAIL_ff00#", "call phonetoken or mail emailtoken"),
        ("See #URL_0f3a9#now", "see urltoken now"),
        ("Visit https://example.com/jobs?id=1 now", "visit urltoken now"),
        ("Visit www.example.com today", "visit urltoken today"),
        ("Write to hr@example.co.in", "write to emailtoken"),
        ("Phone +91 98765 43210.", "phone phonetoken"),
        ("Call (555) 123-4567", "call phonetoken"),
        ("Salary 40000 - 50000 per year", "salary 40000 50000 per year"),
        ("No experience. Not required", "no experience not required"),
        ("Managers wanted", "managers wanted"),
        ("one place.We have", "one place we have"),
    ],
)
def test_p_cvx_cleaning(raw: str, expected: str) -> None:
    assert clean_text_p_cvx(raw) == expected


def test_p_cvx_missing_value_is_empty() -> None:
    assert clean_text_p_cvx(np.nan) == ""


def test_p_cvx_labelled_join(make_postings) -> None:
    df = make_postings([{"title": "Sales Manager", "description": "Sell shoes"}])
    expected = (
        "title: sales manager | company: | description: sell shoes | requirements: | benefits:"
    )
    assert p_cvx(df).tolist() == [expected]


def test_p_cvx_unlabelled_join(make_postings) -> None:
    df = make_postings([{"title": "Sales Manager", "benefits": "Free lunch"}])
    assert p_cvx(df, with_labels=False).tolist() == ["sales manager free lunch"]


def test_fill_missing_categories(make_postings) -> None:
    df = make_postings(
        [
            {"employment_type": "Full-time", "function": np.nan, "industry": "  "},
        ]
    )
    filled = fill_missing_categories(df)
    assert filled.loc[0, "employment_type"] == "Full-time"
    assert filled.loc[0, "function"] == UNKNOWN
    assert filled.loc[0, "industry"] == UNKNOWN
    assert pd.isna(df.loc[0, "function"])  # the input is not modified


def test_build_text_dispatch(make_postings) -> None:
    df = make_postings([{"title": "Not Managers"}])
    assert build_text(df, "p_cv").tolist() == ["manager"]
    assert build_text(df, "p_cvx").iloc[0].startswith("title: not managers |")
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        build_text(df, "p_other")


def test_index_is_preserved(tiny_postings: pd.DataFrame) -> None:
    subset = tiny_postings.iloc[[4, 2]]
    assert list(p_cv(subset).index) == [4, 2]
    assert list(p_cvx(subset).index) == [4, 2]
