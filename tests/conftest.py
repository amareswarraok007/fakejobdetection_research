"""Shared test fixtures: a tiny HAND-WRITTEN set of postings (never real data).

Rows (index: job_id, label):
  0: 1 genuine  Sales Manager
  1: 2 genuine  exact copy of row 0 except job_id      -> exact duplicate
  2: 3 fake     Data Entry Clerk
  3: 4 fake     same text as row 2, different location -> near-duplicate, not exact
  4: 5 genuine  Software Engineer
  5: 6 fake     Payment Processor
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cvx.data.load import EXPECTED_COLUMNS

NAN = np.nan

_SALES = {
    "title": "Sales Manager",
    "location": "US, NY, New York",
    "department": NAN,
    "salary_range": "40000-50000",
    "company_profile": "Acme Ltd sells shoes.",
    "description": "<p>Sell shoes &amp; boots. Call #PHONE_abc123#</p>",
    "requirements": "2 years experience",
    "benefits": NAN,
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1,
    "employment_type": "Full-time",
    "required_experience": "Mid-Senior level",
    "required_education": "Bachelor's Degree",
    "industry": "Retail",
    "function": "Sales",
    "fraudulent": 0,
}
_CLERK = {
    "title": "Data Entry Clerk",
    "location": "US, TX, Houston",
    "description": "Earn money from home! Email #EMAIL_ff00# now",
    "benefits": "Weekly pay",
    "telecommuting": 1,
    "has_company_logo": 0,
    "has_questions": 0,
    "fraudulent": 1,
}
_ENGINEER = {
    "title": "Software Engineer",
    "location": "GB, LND, London",
    "department": "Engineering",
    "company_profile": "We build tools.",
    "description": "Write Python code. Visit https://example.com/jobs",
    "requirements": "Python, SQL",
    "benefits": "Health insurance",
    "has_company_logo": 1,
    "employment_type": "Full-time",
    "required_experience": "Entry level",
    "industry": "Computer Software",
    "function": "Engineering",
    "fraudulent": 0,
}
_PAYMENT = {
    "title": "Payment Processor",
    "location": "US, TX, Houston",
    "salary_range": "1000-2000",
    "description": "No experience needed. Not a scam.",
    "telecommuting": 1,
    "employment_type": "Part-time",
    "industry": "Financial Services",
    "fraudulent": 1,
}


def build_postings(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Make a DataFrame with all 18 EMSCAD columns; unspecified text = NaN, flags = 0."""
    filled = []
    for i, row in enumerate(rows):
        full = {column: NAN for column in EXPECTED_COLUMNS}
        full.update({"job_id": i + 1, "telecommuting": 0, "has_company_logo": 0})
        full.update({"has_questions": 0, "fraudulent": 0})
        full.update(row)
        filled.append(full)
    return pd.DataFrame(filled, columns=list(EXPECTED_COLUMNS))


@pytest.fixture
def make_postings() -> Callable[[list[dict[str, Any]]], pd.DataFrame]:
    """Give tests the ``build_postings`` helper."""
    return build_postings


@pytest.fixture
def tiny_postings() -> pd.DataFrame:
    """The six hand-written postings described in the module docstring."""
    df = build_postings(
        [
            _SALES,
            dict(_SALES),
            _CLERK,
            {**_CLERK, "location": "US, TX, Dallas"},
            _ENGINEER,
            _PAYMENT,
        ]
    )
    df["job_id"] = [1, 2, 3, 4, 5, 6]
    return df
