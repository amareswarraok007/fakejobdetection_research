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


# --- a small corpus for training/run tests --------------------------------------
# Built from a few HAND-WRITTEN phrases. It exists only so the pipeline code can be
# exercised end to end in tests; it is never used for real experiments (CLAUDE.md rule 3).
_FAKE_PHRASES = (
    "earn money from home",
    "registration fee required",
    "weekly pay guaranteed",
    "no experience needed",
    "send your bank details",
    "data entry work",
)
_GENUINE_PHRASES = (
    "software engineer python",
    "join our product team",
    "health insurance and pension",
    "degree in computer science",
    "agile development team",
    "customer success manager",
)
_FILLER = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
).split()
_FUNCTIONS = ("Sales", "Engineering", "Marketing", "Administrative", "Customer Service")


def build_corpus(n_fake: int = 30, n_genuine: int = 60) -> pd.DataFrame:
    """Postings whose wording depends on the label, with a unique ``ref`` word per row."""
    rows = []
    for i in range(n_fake + n_genuine):
        fake = i < n_fake
        phrases = _FAKE_PHRASES if fake else _GENUINE_PHRASES
        filler = " ".join(_FILLER[(i * 7 + k) % len(_FILLER)] for k in range(4))
        rows.append(
            {
                "title": f"{phrases[i % 6].title()} ref{i:03d}",
                "description": f"{phrases[(i + 1) % 6]}. {phrases[(i + 2) % 6]}! {filler}",
                "company_profile": NAN if fake else f"Company {_FILLER[i % 26]} Limited",
                "salary_range": "100-200" if i % 4 == 0 else NAN,
                "employment_type": ("Full-time", "Part-time", "Contract")[i % 3],
                "function": _FUNCTIONS[i % len(_FUNCTIONS)],
                "has_company_logo": 0 if fake else 1,
                "has_questions": i % 2,
                "telecommuting": 1 if fake else 0,
                "fraudulent": int(fake),
            }
        )
    return build_postings(rows)


@pytest.fixture
def corpus() -> pd.DataFrame:
    """90 postings (30 fake, 60 genuine) for pipeline tests."""
    return build_corpus()


_BASE_YAML = """
seed: 42
paths: {data_csv: DATA, runs_dir: RUNS, reports_dir: reports}
tfidf: {ngram_range: [1, 2], min_df: 1, max_df: 1.0, max_features: 500, sublinear_tf: true}
thresholds: [0.5, t_f1, t_r90]
"""
_TINY_CONFIGS = {
    "e0_reproduce": """
experiment_id: e0_reproduce
preprocessing: p_cv
split: {method: random_stratified, ratios: [0.8, 0.1, 0.1]}
models: [lr]
grids: {lr: {C: [0.1, 1.0, 10.0]}}
tuning: {method: validation, scoring: average_precision}
stats: {bootstrap: 20}
""",
    "e1_leakage": """
experiment_id: e1_leakage
preprocessing: p_cv
split: {method: random_stratified, ratios: [0.8, 0.1, 0.1]}
models: [lr]
tuning: {method: validation, scoring: average_precision}
variants:
  - {name: e1a, overrides: {resample: {method: random_oversample, when: before_split}}}
  - {name: e1b, overrides: {resample: {method: smote, when: before_split}}}
  - {name: e1c, overrides: {resample: {method: random_oversample, when: train_only}}}
compare_to: e1c
""",
    "e3_models": """
experiment_id: e3_models
preprocessing: p_cvx
split: {method: grouped_stratified, ratios: [0.6, 0.2, 0.2], near_dup_threshold: 0.95, n_folds: 5}
models: [nb, linear_svm, rf, lr, xgb]
grids: {xgb: {max_depth: [2, 3]}}
tuning: {method: grouped_cv, cv_folds: 3, scoring: average_precision}
stats: {bootstrap: 20, mcnemar: [[lr, xgb]]}
export_models: true
""",
}


@pytest.fixture(scope="session")
def tiny_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run three tiny experiments once per test session; tests read their run folders."""
    from cvx.run import run_experiment

    root = tmp_path_factory.mktemp("tiny_runs")
    data = root / "postings.csv"
    build_corpus().to_csv(data, index=False)
    runs = root / "runs"
    paths = {}
    for name, body in _TINY_CONFIGS.items():
        config = root / f"{name}.yaml"
        text = _BASE_YAML.replace("DATA", data.as_posix()).replace("RUNS", runs.as_posix()) + body
        config.write_text(text, encoding="utf-8")
        paths[name] = run_experiment(config)
    return {"root": runs, "runs": paths, "data": data}
