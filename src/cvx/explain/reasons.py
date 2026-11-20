"""Plain-English reasons from the top SHAP features (PRD §11.5).

Each sentence says what the model looked at and in which direction it pushed. The
wording describes the MODEL ("the model links …"), not a fact about the world:
these are learned patterns from older public postings, not proof.
"""

import math
from typing import Any

import pandas as pd

from cvx.explain.core import TEXT_PREFIX, Explanation
from cvx.features.meta import CATEGORICAL_FEATURES

CAUTION = (
    "These reasons show what the model learned from older public job postings. "
    "They are not proof. Verify the company yourself."
)
TOKEN_WORDS = {
    "urltoken": "a web link",
    "emailtoken": "an email address",
    "phonetoken": "a phone number",
}
FLAG_TEXT = {  # feature: (sentence when 1, sentence when 0)
    "has_company_logo": ("The posting has a company logo", "The posting has no company logo"),
    "has_questions": (
        "The posting has screening questions",
        "The posting has no screening questions",
    ),
    "telecommuting": ("The job is marked as remote", "The job is not marked as remote"),
    "salary_given": ("A salary range is given", "No salary range is given"),
    "company_profile_empty": (
        "The company description is empty",
        "The company description is filled in",
    ),
    "requirements_empty": (
        "The requirements section is empty",
        "The requirements section is filled in",
    ),
    "benefits_empty": ("The benefits section is empty", "The benefits section is filled in"),
}
COUNT_TEXT = {
    "n_url": "web link(s)",
    "n_email": "email address(es)",
    "n_phone": "phone number(s)",
    "money_mentions": "money mention(s)",
    "exclamation_count": "exclamation mark(s)",
}


def _direction(value: float) -> str:
    """'towards fake' or 'towards genuine'."""
    return "towards fake" if value > 0 else "towards genuine"


def _text_reason(ngram: str, value: float) -> str:
    """Sentence for a word or two-word phrase."""
    side = "fake" if value > 0 else "genuine"
    words = ngram.split()
    shown = " ".join(TOKEN_WORDS.get(w, w) for w in words)
    if len(words) == 1 and words[0] in TOKEN_WORDS:
        return f"The model links {shown} in the text with {side} postings."
    kind = "word" if len(words) == 1 else "phrase"
    return f"The model links the {kind} ‘{shown}’ with {side} postings."


def _metadata_reason(name: str, value: float, row: pd.Series) -> str | None:
    """Sentence for a metadata feature, using the posting's own value (None if not shown)."""
    feature = name.split("__", 1)[1]
    if feature in FLAG_TEXT:
        sentence = FLAG_TEXT[feature][0 if float(row[feature]) == 1 else 1]
    elif feature in COUNT_TEXT:
        sentence = f"The posting contains {int(row[feature])} {COUNT_TEXT[feature]}"
    elif feature == "uppercase_ratio":
        sentence = f"Capital letters make up {float(row[feature]):.0%} of the letters"
    elif feature == "log_text_length":
        sentence = (
            f"The posting text is about {round(math.expm1(float(row[feature])))} characters long"
        )
    else:
        category = _category_of(feature, row)
        if category is None:
            return None
        sentence = category
    return f"{sentence}; this pushes the score {_direction(value)}."


def _category_of(feature: str, row: pd.Series) -> str | None:
    """'Employment type is Full-time' if the one-hot feature is the posting's own value."""
    for column in CATEGORICAL_FEATURES:
        prefix = f"{column}_"
        if feature.startswith(prefix):
            value = feature[len(prefix) :]
            if str(row.get(column)) == value:
                return f"{column.replace('_', ' ').capitalize()} is ‘{value}’"
            return None
    return None


def reasons(explanation: Explanation, row: pd.DataFrame | None = None, top: int = 5) -> list[str]:
    """Up to ``top`` sentences for the features with the largest effect.

    ``row``: the one-row model-input table (needed to phrase metadata reasons).
    """
    values = row.iloc[0] if row is not None else pd.Series(dtype=object)
    ranked = sorted(explanation.features.items(), key=lambda item: (-abs(item[1]), item[0]))
    sentences: list[str] = []
    for name, value in ranked:
        if len(sentences) == top or value == 0:
            break
        is_metadata = "__" in name and not name.startswith(TEXT_PREFIX)
        sentence = (
            _metadata_reason(name, value, values) if is_metadata else _text_reason(name, value)
        )
        if sentence is not None and sentence not in sentences:
            sentences.append(sentence)
    return sentences


def format_reasons(items: list[str]) -> dict[str, Any]:
    """Reasons plus the fixed caution text, ready for the app or a JSON file."""
    return {"reasons": items, "caution": CAUTION}
