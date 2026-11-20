"""Model input table and feature transformer (text-only or text + metadata).

``make_model_input`` turns raw postings into the table every model reads:
a ``text`` column (P-CV or P-CVX) plus, optionally, the metadata columns.
``make_preprocessor`` builds the scikit-learn ColumnTransformer that turns that
table into numbers. Training and the app both use these two functions.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cvx.config import TfidfConfig
from cvx.data import clean
from cvx.features.meta import CATEGORICAL_FEATURES, NUMERIC_FEATURES, metadata_frame
from cvx.features.text import make_tfidf

TEXT_COLUMN = "text"
# PRD §7.2: "function" keeps its top 20 values plus one "Other" bucket.
FUNCTION_MAX_CATEGORIES = 21
# E7a audit columns (location, industry): values seen fewer than 5 times share one bucket.
AUDIT_MIN_FREQUENCY = 5


def make_model_input(
    df: pd.DataFrame, preprocessing: str, metadata: bool, extra_categories: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Build the model input table from raw postings.

    Input: raw postings, ``p_cv`` or ``p_cvx``, whether to add metadata, and optional
    extra category columns for the shortcut audit (E7a: location, industry).
    Output: DataFrame (same index) with ``text`` and, if asked, the other columns.
    """
    if preprocessing == "p_cvx":
        fields = clean.clean_fields_p_cvx(df)
        text = clean.join_p_cvx(fields, with_labels=True)
    else:
        fields = None
        text = clean.build_text(df, preprocessing)
    table = pd.DataFrame({TEXT_COLUMN: text}, index=df.index)
    if metadata:
        table = table.join(metadata_frame(df, cleaned_fields=fields))
    if extra_categories:
        filled = clean.fill_missing_categories(df)
        for column in extra_categories:
            table[column] = filled[column].astype(str)
    return table


def make_preprocessor(
    tfidf: TfidfConfig, metadata: bool, extra_categories: tuple[str, ...] = ()
) -> ColumnTransformer:
    """Create the unfitted feature transformer.

    Text -> TF-IDF. With metadata: numeric columns -> ``StandardScaler(with_mean=False)``
    (keeps the matrix sparse and values non-negative, which Naive Bayes needs);
    categories -> one-hot, with ``function`` limited to its top 20 values plus Other.
    """
    parts: list[tuple[str, object, object]] = [("text", make_tfidf(tfidf), TEXT_COLUMN)]
    if metadata:
        other_categories = [c for c in CATEGORICAL_FEATURES if c != "function"]
        parts += [
            ("numeric", StandardScaler(with_mean=False), list(NUMERIC_FEATURES)),
            ("category", OneHotEncoder(handle_unknown="ignore"), other_categories),
            (
                "function",
                OneHotEncoder(
                    max_categories=FUNCTION_MAX_CATEGORIES,
                    handle_unknown="infrequent_if_exist",
                ),
                ["function"],
            ),
        ]
    if extra_categories:
        audit = OneHotEncoder(
            min_frequency=AUDIT_MIN_FREQUENCY, handle_unknown="infrequent_if_exist"
        )
        parts.append(("audit", audit, list(extra_categories)))
    # sparse_threshold=1.0: always return a sparse matrix (TF-IDF is mostly zeros).
    return ColumnTransformer(parts, sparse_threshold=1.0)
