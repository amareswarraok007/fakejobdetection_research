"""LIME for one posting, with metadata held fixed (PRD E5, §11.5).

LIME removes random words from the posting text, asks the model for a score each
time, and fits a small linear model to see which words matter. Here:
- only the ``text`` column changes; every metadata column keeps the posting's value,
- EXCEPT the counts that are computed from the cleaned text (``log_text_length``,
  ``n_url``, ``n_email``, ``n_phone``), which are recomputed from each edited text so
  they stay consistent with it (PRD M4 criterion). Counts taken from the raw text
  before cleaning (exclamations, capitals, money) cannot be recomputed and stay fixed.
``predict_texts`` is also used by the faithfulness tests, so both use the same rules.
"""

import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from cvx.data.clean import FIELD_LABELS
from cvx.explain.core import Explanation
from cvx.features.build import TEXT_COLUMN
from cvx.features.meta import COUNT_TOKENS

_LABEL = re.compile(r"\b(?:" + "|".join(FIELD_LABELS) + r"):")
_WORD = re.compile(r"\w+")
TEXT_COUNT_COLUMNS = ("log_text_length", *COUNT_TOKENS)


def unlabelled_tokens(text: str) -> list[str]:
    """Words of a P-CVX text without the field labels (``title:``, ``company:`` …)."""
    return _WORD.findall(_LABEL.sub(" ", text))


def text_counts(text: str) -> dict[str, float]:
    """Text-based counts recomputed from a (possibly edited) P-CVX text."""
    tokens = unlabelled_tokens(text)
    counts = {"log_text_length": math.log1p(len(" ".join(tokens)))}
    for column, token in COUNT_TOKENS.items():
        counts[column] = float(sum(t == token for t in tokens))
    return counts


def rows_for_texts(row: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
    """Copies of one model-input row with new texts; text-based counts recomputed."""
    rows = pd.concat([row.iloc[[0]]] * len(texts), ignore_index=True)
    rows[TEXT_COLUMN] = list(texts)
    if all(column in rows.columns for column in TEXT_COUNT_COLUMNS):
        counts = pd.DataFrame([text_counts(t) for t in texts])
        for column in TEXT_COUNT_COLUMNS:
            rows[column] = counts[column].to_numpy()
    return rows


def predict_texts(estimator: Any, row: pd.DataFrame, texts: list[str]) -> np.ndarray:
    """Probability of fake for each edited text, everything else as in ``row``."""
    return np.asarray(estimator.predict_proba(rows_for_texts(row, texts)))[:, 1]


def explain_lime(
    estimator: Any,
    row: pd.DataFrame,
    num_samples: int = 1000,
    num_features: int = 20,
    seed: int = 42,
) -> tuple[Explanation, float]:
    """LIME explanation of one posting (``row`` = a one-row model-input table).

    Output: the ``Explanation`` (word weights towards fake) and the seconds it took.
    ``bow=True`` means removing a word removes every occurrence of it.
    """
    start = time.perf_counter()
    explainer = LimeTextExplainer(class_names=["genuine", "fake"], bow=True, random_state=seed)

    def classifier(texts: list[str]) -> np.ndarray:
        """LIME's callback: two-column probabilities (genuine, fake)."""
        fake = predict_texts(estimator, row, list(texts))
        return np.column_stack([1 - fake, fake])

    text = str(row[TEXT_COLUMN].iloc[0])
    result = explainer.explain_instance(
        text, classifier, labels=(1,), num_features=num_features, num_samples=num_samples
    )
    weights = {word: float(w) for word, w in result.as_list(label=1)}
    score = float(predict_texts(estimator, row, [text])[0])
    return Explanation(
        "lime", score, None, dict(weights), dict(weights)
    ), time.perf_counter() - start
