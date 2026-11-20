"""Shared pieces for explanations: the Explanation record, per-word scores, global ranking.

SHAP gives one number per *model feature*. Model features are TF-IDF words and
two-word phrases (named ``text__<words>``) plus metadata features. For people and
for the faithfulness tests we need one number per *word*, so (PRD E6):
- a one-word feature gives its whole score to that word;
- a two-word phrase splits its score equally between its two words.
Only words that are actually in the posting are listed; the summed effect of
vocabulary words that are absent is kept separately so nothing is lost.
Positive scores push towards FAKE, negative towards GENUINE.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

TEXT_PREFIX = "text__"
ABSENT_WORDS = "words not in posting"


@dataclass
class Explanation:
    """One posting's explanation.

    ``score``: model probability of fake. ``base_value``: SHAP expected value (log-odds),
    None for LIME. ``features``: contribution of each feature present in the posting
    (n-grams and metadata). ``words``: per-word contribution. ``absent_total``: summed
    SHAP of vocabulary features absent from the posting (0 for LIME).
    """

    method: str
    score: float
    base_value: float | None
    features: dict[str, float] = field(default_factory=dict)
    words: dict[str, float] = field(default_factory=dict)
    absent_total: float = 0.0


def feature_names(estimator: Any) -> np.ndarray:
    """Readable feature names of a fitted pipeline's ``features`` step."""
    return np.asarray(estimator.named_steps["features"].get_feature_names_out())


def transform_features(estimator: Any, X: Any) -> sparse.csr_matrix:
    """Turn model input rows into the model's feature matrix (sparse)."""
    return sparse.csr_matrix(estimator.named_steps["features"].transform(X))


def words_from_ngrams(ngram_scores: dict[str, float]) -> dict[str, float]:
    """Per-word scores: a phrase's score is split equally between its words."""
    words: dict[str, float] = defaultdict(float)
    for ngram, value in ngram_scores.items():
        tokens = ngram.split()
        for token in tokens:
            words[token] += value / len(tokens)
    return dict(words)


def explanation_from_row(
    method: str,
    names: np.ndarray,
    phi: np.ndarray,
    x_row: sparse.csr_matrix,
    score: float,
    base_value: float | None,
) -> Explanation:
    """Build an ``Explanation`` from one row of SHAP values and that row's features.

    A text feature counts as present when its feature value is non-zero; metadata
    features are always listed (their value is part of the posting).
    """
    present = set(x_row.indices.tolist())
    is_text = np.char.startswith(names.astype(str), TEXT_PREFIX)
    features: dict[str, float] = {}
    ngrams: dict[str, float] = {}
    for index in np.flatnonzero(~is_text):
        features[str(names[index])] = float(phi[index])
    text_present = [i for i in present if is_text[i]]
    for index in text_present:
        name = str(names[index])[len(TEXT_PREFIX) :]
        features[name] = ngrams[name] = float(phi[index])
    absent = float(phi[is_text].sum() - sum(ngrams.values()))
    return Explanation(method, score, base_value, features, words_from_ngrams(ngrams), absent)


def top_words(explanation: Explanation, k: int, toward: int | None = None) -> list[str]:
    """The ``k`` most important words.

    ``toward=1``: largest push towards fake; ``toward=0``: largest push towards genuine;
    ``None``: largest absolute effect. Ties keep alphabetical order (deterministic).
    """

    def importance(item: tuple[str, float]) -> float:
        """Sort key: smaller means more important for the requested direction."""
        value = item[1]
        if toward is None:
            return -abs(value)
        return -value if toward == 1 else value

    return [word for word, _ in sorted(sorted(explanation.words.items()), key=importance)[:k]]


def global_ranking(names: np.ndarray, phi: np.ndarray, top: int = 30) -> list[dict[str, Any]]:
    """Features ranked by mean absolute SHAP value over many postings (global importance)."""
    mean_abs = np.abs(np.asarray(phi)).mean(axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:top]
    return [{"feature": str(names[i]), "mean_abs_shap": float(mean_abs[i])} for i in order]
