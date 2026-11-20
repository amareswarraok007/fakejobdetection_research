"""Turn the app's form into the same input table the models were trained on (PRD §11.2).

Two modes:
- ``full``: a full posting (title*, description*, company description, requirements,
  benefits, salary, dropdowns, yes/no questions) -> text + metadata model.
- ``text``: one pasted message (WhatsApp, email …) -> text-only model. The message is
  placed in the ``description`` field; the other fields stay empty.
The same cleaning and feature code as training is used (``make_model_input``).
"""

from typing import Any

import numpy as np
import pandas as pd

from cvx.data.load import EXPECTED_COLUMNS
from cvx.features.build import make_model_input

MODES = ("full", "text")
MAX_CHARS = 20_000
TEXT_INPUTS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "salary_range",
)
CATEGORY_INPUTS = ("employment_type", "required_experience", "required_education", "function")
FLAG_INPUTS = ("has_company_logo", "has_questions", "telecommuting")
REQUIRED = {"full": ("title", "description"), "text": ("message",)}


class InputError(ValueError):
    """The form is incomplete or too long; the message is shown to the user."""


def _value(form: dict[str, Any], key: str) -> str:
    """A stripped text value from the form ('' when missing)."""
    return str(form.get(key) or "").strip()


def validate_form(form: dict[str, Any], mode: str) -> list[str]:
    """Problems with the form (empty list when it is fine)."""
    if mode not in MODES:
        return [f"Unknown input mode {mode!r}."]
    problems = [
        f"Please fill in '{key.replace('_', ' ')}'."
        for key in REQUIRED[mode]
        if not _value(form, key)
    ]
    total = sum(len(str(v)) for v in form.values() if isinstance(v, str))
    if total > MAX_CHARS:
        problems.append(f"The input is longer than {MAX_CHARS:,} characters.")
    return problems


def build_posting(form: dict[str, Any], mode: str) -> pd.DataFrame:
    """One-row raw posting with the 18 EMSCAD columns (missing values as NaN / Unknown)."""
    row: dict[str, Any] = {column: np.nan for column in EXPECTED_COLUMNS}
    row.update(job_id=0, fraudulent=0)
    for flag in FLAG_INPUTS:
        row[flag] = 1 if (mode == "full" and form.get(flag)) else 0
    if mode == "text":
        row["description"] = _value(form, "message")
    else:
        for key in TEXT_INPUTS:
            row[key] = _value(form, key) or np.nan
        for key in CATEGORY_INPUTS:
            choice = _value(form, key)
            row[key] = choice if choice and choice != "Unknown" else np.nan
    return pd.DataFrame([row], columns=list(EXPECTED_COLUMNS))


def build_input(form: dict[str, Any], mode: str) -> pd.DataFrame:
    """Validated form -> one-row model input (text + metadata; text-only models use text)."""
    problems = validate_form(form, mode)
    if problems:
        raise InputError(" ".join(problems))
    return make_model_input(build_posting(form, mode), "p_cvx", metadata=True).reset_index(
        drop=True
    )


def display_text(form: dict[str, Any], mode: str) -> str:
    """The text shown (with highlights) on the result: the message, or the filled fields."""
    if mode == "text":
        return _value(form, "message")
    parts = [
        _value(form, key)
        for key in ("title", "company_profile", "description", "requirements", "benefits")
    ]
    return "\n\n".join(part for part in parts if part)


def rules_text(form: dict[str, Any], mode: str) -> str:
    """All free text the safety rules should read (includes company name and salary)."""
    extra = [_value(form, "company_name"), _value(form, "salary_range")] if mode == "full" else []
    return "\n".join(part for part in [display_text(form, mode), *extra] if part)
