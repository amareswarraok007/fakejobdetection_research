"""PRD M4 check: the LIME wrapper keeps metadata fixed and recomputes text-based counts."""

from typing import Any

import pytest

from cvx.explain.lime_text import (
    TEXT_COUNT_COLUMNS,
    explain_lime,
    predict_texts,
    rows_for_texts,
    text_counts,
    unlabelled_tokens,
)


def test_unlabelled_tokens() -> None:
    text = "title: data entry | company: | description: email emailtoken now"
    assert unlabelled_tokens(text) == ["data", "entry", "email", "emailtoken", "now"]


def test_counts_of_unchanged_text_equal_the_metadata(fitted_models: dict[str, Any]) -> None:
    table = fitted_models["table"]
    for i in range(5):
        row = table.iloc[i]
        counts = text_counts(row["text"])
        for column in TEXT_COUNT_COLUMNS:
            assert counts[column] == pytest.approx(row[column])


def test_edited_text_recomputes_counts_and_keeps_other_metadata(
    fitted_models: dict[str, Any],
) -> None:
    row = fitted_models["table"].iloc[[0]]
    edited = rows_for_texts(row, ["title: x | description: urltoken urltoken", row["text"].iloc[0]])
    assert (
        edited.loc[0, "n_url"] == 2
        and edited.loc[0, "log_text_length"] < row["log_text_length"].iloc[0]
    )
    for column in ("has_company_logo", "exclamation_count", "employment_type", "function"):
        assert (edited[column] == row[column].iloc[0]).all()  # held fixed
    assert edited.loc[1, "log_text_length"] == pytest.approx(row["log_text_length"].iloc[0])


def test_text_only_model_rows(fitted_models: dict[str, Any]) -> None:
    row = fitted_models["table"][["text"]].iloc[[0]]
    scores = predict_texts(fitted_models["models"]["lr_text"], row, ["a b", "c d", "e"])
    assert scores.shape == (3,)


@pytest.mark.parametrize("name", ["lr_full", "xgb_text"])
def test_explain_lime_is_reproducible(fitted_models: dict[str, Any], name: str) -> None:
    row = fitted_models["table"].iloc[[0]]
    model = fitted_models["models"][name]
    first, seconds = explain_lime(model, row, num_samples=200, num_features=5, seed=42)
    second, _ = explain_lime(model, row, num_samples=200, num_features=5, seed=42)
    assert first.words == second.words and len(first.words) == 5
    assert first.method == "lime" and seconds > 0
    assert first.score == pytest.approx(model.predict_proba(row)[0, 1])
