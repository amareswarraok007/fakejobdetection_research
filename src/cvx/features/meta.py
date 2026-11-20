"""Metadata features (PRD Section 7.2), computed from the raw postings.

Two kinds of text-based counts, as agreed with the owner:
- From the cleaned P-CVX text without labels: ``log_text_length``, ``n_url``,
  ``n_email``, ``n_phone``. (In M4 the LIME wrapper can recompute these from edited text.)
- From the text after HTML stripping and contact masking but BEFORE lowercasing:
  ``exclamation_count``, ``uppercase_ratio``, ``money_mentions``. Masking first
  stops the dataset's ``#URL_…#`` placeholders from adding capital letters.
"""

import math
import re

import pandas as pd

from cvx.data import clean
from cvx.data.load import BINARY_FIELDS, CATEGORY_FIELDS, SALARY_FIELD, TEXT_FIELDS, is_missing

EMPTY_FLAGS = {
    "company_profile_empty": "company_profile",
    "requirements_empty": "requirements",
    "benefits_empty": "benefits",
}
COUNT_TOKENS = {"n_url": "urltoken", "n_email": "emailtoken", "n_phone": "phonetoken"}
NUMERIC_FEATURES = (
    *BINARY_FIELDS,
    "salary_given",
    *EMPTY_FLAGS,
    "log_text_length",
    *COUNT_TOKENS,
    "money_mentions",
    "exclamation_count",
    "uppercase_ratio",
)
CATEGORICAL_FEATURES = CATEGORY_FIELDS
# Money patterns from PRD §7.2: "$", "₹", "rs.", "inr", "per week", "per day".
_MONEY = re.compile(r"\$|₹|\brs\.|\binr\b|\bper week\b|\bper day\b", re.IGNORECASE)


def count_money(text: str) -> int:
    """Number of money patterns in ``text`` (case-insensitive)."""
    return len(_MONEY.findall(text))


def uppercase_ratio(text: str) -> float:
    """Capital letters divided by all letters; 0.0 when there are no letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)


def masked_raw_text(df: pd.DataFrame) -> pd.Series:
    """Five raw fields, each HTML-stripped and contact-masked (case kept), joined by spaces."""

    def one_field(value: object) -> str:
        """HTML-stripped, contact-masked text of one field, case kept."""
        return clean.mask_contacts(clean.strip_html(clean.as_text(value)))

    parts = pd.DataFrame({field: df[field].map(one_field) for field in TEXT_FIELDS})
    return parts.agg(" ".join, axis=1)


def _token_counts(unlabelled: pd.Series) -> pd.DataFrame:
    """Length and placeholder-token counts from the cleaned text without labels."""
    tokens = unlabelled.str.split()
    out = pd.DataFrame(index=unlabelled.index)
    out["log_text_length"] = unlabelled.str.len().map(lambda n: math.log1p(n))
    for name, token in COUNT_TOKENS.items():
        out[name] = tokens.map(lambda words, t=token: sum(w == t for w in words))
    return out


def metadata_frame(df: pd.DataFrame, cleaned_fields: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute every metadata feature for every row.

    Input: raw postings; optionally the output of ``clean.clean_fields_p_cvx`` to avoid
    cleaning twice. Output: DataFrame with ``NUMERIC_FEATURES`` (floats) and
    ``CATEGORICAL_FEATURES`` (strings, missing -> ``Unknown``), same index as ``df``.
    """
    fields = cleaned_fields if cleaned_fields is not None else clean.clean_fields_p_cvx(df)
    out = pd.DataFrame(index=df.index)
    for column in BINARY_FIELDS:
        out[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float)
    out["salary_given"] = (~is_missing(df[SALARY_FIELD])).astype(float)
    for name, field in EMPTY_FLAGS.items():
        out[name] = (fields[field] == "").astype(float)
    out = out.join(_token_counts(clean.join_p_cvx(fields, with_labels=False)))
    raw = masked_raw_text(df)
    out["money_mentions"] = raw.map(count_money).astype(float)
    out["exclamation_count"] = raw.str.count("!").astype(float)
    out["uppercase_ratio"] = raw.map(uppercase_ratio)
    categories = clean.fill_missing_categories(df)
    for column in CATEGORICAL_FEATURES:
        out[column] = categories[column].astype(str)
    return out[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
