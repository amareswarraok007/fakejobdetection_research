"""Load the EMSCAD CSV (Kaggle "Real / Fake Job Posting Prediction") and check its columns.

Missing values are kept as NaN here on purpose: profiling needs to see them, and
each preprocessing pipeline (``clean.py``) decides how to fill them.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KAGGLE_URL = "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction"

ID_COLUMN = "job_id"
LABEL_COLUMN = "fraudulent"
# The five free-text fields that both P-CV and P-CVX join together.
TEXT_FIELDS = ("title", "company_profile", "description", "requirements", "benefits")
BINARY_FIELDS = ("telecommuting", "has_company_logo", "has_questions")
CATEGORY_FIELDS = ("employment_type", "required_experience", "required_education", "function")
# Used only by the shortcut audit (E7) and the data profile.
AUDIT_FIELDS = ("location", "department", "industry")
SALARY_FIELD = "salary_range"

EXPECTED_COLUMNS = (
    ID_COLUMN,
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    LABEL_COLUMN,
)


class DataFileMissingError(FileNotFoundError):
    """Raised when the dataset CSV is not where the config says it is."""


def load_postings(csv_path: str | Path) -> pd.DataFrame:
    """Read the EMSCAD CSV and check it has the 18 expected columns and a 0/1 label.

    Input: path to ``fake_job_postings.csv``.
    Output: DataFrame with one row per posting; text columns may contain NaN.
    Raises ``DataFileMissingError`` if the file is absent, ``ValueError`` if it is malformed.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise DataFileMissingError(
            f"Dataset not found at '{path}'.\n"
            f"Download 'fake_job_postings.csv' from {KAGGLE_URL}\n"
            f"and save it as '{path}' (the 'data/' folder is not committed to git)."
        )
    df = pd.read_csv(path)
    _check_columns(df, path)
    _check_label(df, path)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def _check_columns(df: pd.DataFrame, path: Path) -> None:
    """Fail clearly if any expected column is missing (e.g. a wrong or edited file)."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"'{path}' is missing expected column(s): {missing}")


def _check_label(df: pd.DataFrame, path: Path) -> None:
    """The label must be 0 (genuine) or 1 (fake) on every row."""
    bad = ~df[LABEL_COLUMN].isin([0, 1])
    if bad.any():
        raise ValueError(f"'{path}' has {int(bad.sum())} row(s) where '{LABEL_COLUMN}' is not 0/1.")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks so big files fit in memory.

    Used to record exactly which dataset file produced a result.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(series: pd.Series) -> pd.Series:
    """True where a value is NaN or an empty / whitespace-only string."""
    as_text = series.astype("string").str.strip()
    return (series.isna() | (as_text == "").fillna(True)).astype(bool)
