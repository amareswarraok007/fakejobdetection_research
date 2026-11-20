"""Text cleaning: P-CV (copy of Career Verify) and P-CVX (this project). PRD Section 6.4.

P-CV  — lowercase; keep letters only (removes numbers, punctuation, special
        characters); remove NLTK English stop words; NLTK WordNet lemmatiser;
        the five text fields joined with spaces. Used only in E0, E1, E2a.
P-CVX — strip HTML; mask URLs, emails and phone numbers as ``urltoken``,
        ``emailtoken``, ``phonetoken``; lowercase; remove punctuation; keep every
        word (so "no" and "not" stay); no lemmatisation so explanation words stay
        readable; the five fields joined with labels: ``title: … | company: …``.

This is the ONLY place cleaning code lives. Training and the app both call it.
"""

import html
import logging
import re
from collections.abc import Callable
from functools import lru_cache

import pandas as pd

from cvx.data.load import AUDIT_FIELDS, CATEGORY_FIELDS, TEXT_FIELDS

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
# Field labels used in the P-CVX joined text, in TEXT_FIELDS order.
FIELD_LABELS = ("title", "company", "description", "requirements", "benefits")

# --- patterns shared by P-CVX -------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
# The dataset authors already replaced links, emails and phones with placeholders
# such as "#URL_0f3a...#". They can be glued to words, hence no word boundaries.
_DATASET_PLACEHOLDERS = (
    (re.compile(r"#URL_[^#\s]*#"), " urltoken "),
    (re.compile(r"#EMAIL_[^#\s]*#"), " emailtoken "),
    (re.compile(r"#PHONE_[^#\s]*#"), " phonetoken "),
)
# Real links/emails/phones appear in text pasted into the app; mask them the same way.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# Phone candidate: optional "+", then digits with at most two separator characters
# between them. Three-character gaps such as "40000 - 50000" therefore do not match.
_PHONE_CANDIDATE = re.compile(r"(?<![\w+])\+?\(?\d(?:[\s.\-()]{0,2}\d){7,}")
_PHONE_MIN_DIGITS, _PHONE_MAX_DIGITS = 10, 15
_NON_WORD = re.compile(r"[\W_]+")
_NON_LETTER = re.compile(r"[^a-z]+")


def as_text(value: object) -> str:
    """Turn NaN/None into an empty string and anything else into ``str``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


# --- P-CV ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _nltk_tools() -> tuple[frozenset[str], Callable[[str], str]]:
    """Load NLTK stop words and a cached WordNet lemmatiser once per process."""
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        stop_words = frozenset(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")  # forces WordNet to load now, not mid-run
    except LookupError as error:
        raise RuntimeError(
            "NLTK data for P-CV is missing. Run once:\n"
            "    python -m nltk.downloader stopwords wordnet"
        ) from error
    # The same words repeat thousands of times, so caching saves a lot of time.
    return stop_words, lru_cache(maxsize=None)(lemmatizer.lemmatize)


def clean_text_p_cv(text: object) -> str:
    """Clean one text value the Career Verify way.

    Input: raw text (NaN allowed). Output: lowercase lemmatised words separated by
    single spaces, with stop words, numbers, punctuation and special characters removed.
    """
    stop_words, lemmatize = _nltk_tools()
    letters_only = _NON_LETTER.sub(" ", as_text(text).lower())
    words = [w for w in letters_only.split() if w not in stop_words]
    return " ".join(lemmatize(w) for w in words)


def p_cv(df: pd.DataFrame) -> pd.Series:
    """Build the P-CV text for every row: the five cleaned fields joined by spaces.

    Input: postings DataFrame. Output: Series of strings with the same index.
    """
    cleaned = [df[field].map(clean_text_p_cv) for field in TEXT_FIELDS]
    joined = pd.concat(cleaned, axis=1).agg(" ".join, axis=1)
    return joined.str.split().str.join(" ").rename("text")


# --- P-CVX ---------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Remove HTML tags and turn entities like ``&amp;`` back into characters."""
    return html.unescape(_HTML_TAG.sub(" ", text))


def _mask_phone(match: re.Match[str]) -> str:
    """Replace a phone candidate only if it has a realistic number of digits."""
    digits = sum(ch.isdigit() for ch in match.group(0))
    if _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS:
        return " phonetoken "
    return match.group(0)


def mask_contacts(text: str) -> str:
    """Replace dataset placeholders and real URLs, emails and phone numbers with tokens."""
    for pattern, token in _DATASET_PLACEHOLDERS:
        text = pattern.sub(token, text)
    text = _EMAIL.sub(" emailtoken ", text)  # before URLs: emails contain dots too
    text = _URL.sub(" urltoken ", text)
    return _PHONE_CANDIDATE.sub(_mask_phone, text)


def clean_text_p_cvx(text: object) -> str:
    """Clean one text value the CVX way.

    Input: raw text (NaN allowed). Output: lowercase words (letters and digits)
    separated by single spaces; HTML removed; contacts masked; no words dropped.
    """
    text = mask_contacts(strip_html(as_text(text))).lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


def clean_fields_p_cvx(df: pd.DataFrame) -> pd.DataFrame:
    """Clean each of the five text fields separately with P-CVX.

    Output: DataFrame with columns ``TEXT_FIELDS`` and the same index as ``df``.
    Used for the joined text and for metadata such as ``benefits_empty``.
    """
    return pd.DataFrame({field: df[field].map(clean_text_p_cvx) for field in TEXT_FIELDS})


def join_p_cvx(fields: pd.DataFrame, with_labels: bool = True) -> pd.Series:
    """Join cleaned fields (from ``clean_fields_p_cvx``) into one text per row."""
    if with_labels:
        # rstrip so an empty field reads "company:" rather than "company: ".
        parts = [
            (label + ": " + fields[field]).str.rstrip()
            for label, field in zip(FIELD_LABELS, TEXT_FIELDS, strict=True)
        ]
        return pd.concat(parts, axis=1).agg(" | ".join, axis=1).rename("text")
    joined = fields[list(TEXT_FIELDS)].agg(" ".join, axis=1)
    return joined.str.split().str.join(" ").rename("text")


def p_cvx(df: pd.DataFrame, with_labels: bool = True) -> pd.Series:
    """Build the P-CVX text for every row.

    With labels (default, used for features): ``title: … | company: … | description: …``.
    Without labels: the cleaned fields joined by spaces. Near-duplicate detection uses
    this form, because the five label words appear in every posting and would make
    short postings look similar to each other.
    Output: Series of strings with the same index as ``df``.
    """
    return join_p_cvx(clean_fields_p_cvx(df), with_labels=with_labels)


def fill_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where empty category and audit columns become ``Unknown`` (P-CVX)."""
    out = df.copy()
    for column in (*CATEGORY_FIELDS, *AUDIT_FIELDS):
        values = out[column].astype("string").str.strip()
        out[column] = values.mask(values.isna() | (values == ""), UNKNOWN).astype(object)
    return out


PREPROCESSORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {"p_cv": p_cv, "p_cvx": p_cvx}


def build_text(df: pd.DataFrame, method: str) -> pd.Series:
    """Run the preprocessing named in the config (``p_cv`` or ``p_cvx``) on every row."""
    if method not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing {method!r}. Use one of {list(PREPROCESSORS)}.")
    logger.info("Cleaning %d postings with %s", len(df), method)
    return PREPROCESSORS[method](df)
