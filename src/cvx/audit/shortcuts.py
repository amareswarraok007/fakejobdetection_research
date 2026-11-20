"""Shortcut audit helpers (PRD E7).

A *shortcut* is a feature that works on this dataset by accident (for example a city
name that happens to appear in many fake postings) and will not generalise.
- ``place_words``: words that occur in the dataset's ``location`` values.
- ``flag_tokens``: global top-30 SHAP words that are place words (candidates to mask).
  This is an automatic first pass; the owner reviews the top-30 list by hand as well
  and can add tokens with ``xai.mask_tokens`` in the config.
- ``mask_text``: replace flagged words with ``maskedtoken`` (E7b re-trains on this).
"""

import re
from typing import Any

import pandas as pd

from cvx.data.load import is_missing
from cvx.explain.core import TEXT_PREFIX

MASK = "maskedtoken"
MIN_PLACE_WORD = 3  # skip 2-letter codes such as 'us', which are also English words


def place_words(postings: pd.DataFrame, stop_words: frozenset[str] = frozenset()) -> set[str]:
    """Lower-case words (3+ letters, not stop words) found in the ``location`` column."""
    values = postings.loc[~is_missing(postings["location"]), "location"].astype(str)
    words: set[str] = set()
    for value in values:
        words.update(w for w in re.findall(r"[a-z]+", value.lower()) if len(w) >= MIN_PLACE_WORD)
    return words - set(stop_words)


def flag_tokens(ranking: list[dict[str, Any]], places: set[str], extra: list[str]) -> list[str]:
    """Top-ranked text words that are place words, plus the config's extra tokens."""
    flagged: dict[str, None] = {}
    for item in ranking:
        name = item["feature"]
        if name.startswith(TEXT_PREFIX):
            for token in name[len(TEXT_PREFIX) :].split():
                if token in places:
                    flagged.setdefault(token, None)
    for token in extra:
        flagged.setdefault(token.lower(), None)
    return list(flagged)


def mask_text(texts: pd.Series, tokens: list[str]) -> pd.Series:
    """Replace every whole-word occurrence of ``tokens`` with ``maskedtoken``."""
    if not tokens:
        return texts.copy()
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, tokens)) + r")(?!\w)")
    return texts.map(lambda text: pattern.sub(MASK, text))


def audit_share(ranking: list[dict[str, Any]], prefix: str = "audit__") -> float:
    """Share of the ranked features that come from the audit columns (location, industry)."""
    if not ranking:
        return 0.0
    return sum(item["feature"].startswith(prefix) for item in ranking) / len(ranking)
