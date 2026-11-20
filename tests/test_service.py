"""Tests for cvx.schema and cvx.service with a tiny model store trained inside the tests."""

import time
from pathlib import Path

import pytest

from cvx import service
from cvx.schema import InputError, build_input, build_posting, display_text, rules_text

FULL = {
    "title": "Data Entry Clerk",
    "company_name": "Acme",
    "description": "Earn money from home! Pay the registration fee by UPI. Visit www.x.com",
    "salary_range": "100-200",
    "employment_type": "Part-time",
    "function": "Unknown",
    "has_company_logo": False,
    "has_questions": False,
    "telecommuting": True,
    "recruiter_email": "hr@gmail.com",
    "company_website": "acme.com",
}
TEXT = {"message": "Weekly pay guaranteed, contact on WhatsApp <b>now</b>"}


def test_schema_modes() -> None:
    posting = build_posting(FULL, "full")
    assert posting.loc[0, "telecommuting"] == 1 and posting.loc[0, "has_company_logo"] == 0
    assert posting.isna().loc[0, "function"]  # "Unknown" -> missing -> Unknown again later
    text_row = build_posting(TEXT, "text")
    assert text_row.loc[0, "description"] == TEXT["message"] and text_row.isna().loc[0, "title"]
    row = build_input(FULL, "full")
    assert row.loc[0, "n_url"] == 1 and row.loc[0, "text"].startswith("title: data entry clerk")
    assert "Acme" in rules_text(FULL, "full") and display_text(TEXT, "text") == TEXT["message"]


def test_schema_rejects_bad_input() -> None:
    with pytest.raises(InputError, match="title"):
        build_input({"description": "x"}, "full")
    with pytest.raises(InputError, match="20,000"):
        build_input({"message": "x" * 20_001}, "text")


@pytest.fixture(scope="module")
def store(tiny_store: Path):
    return service.open_store(str(tiny_store))


def test_choose_model(store) -> None:
    assert service.choose_model(store, "full", "LR") == "lr_full"
    assert service.choose_model(store, "text", "XGBoost") == "xgb_text"
    assert service.choose_model(store, "text", "Auto") in ("lr_text", "xgb_text")


@pytest.mark.parametrize("choice", ["LR", "XGBoost"])
def test_analyze_full(store, choice: str) -> None:
    start = time.perf_counter()
    result = service.analyze(FULL, "full", choice, store)
    assert time.perf_counter() - start < 2.0  # PRD: SHAP result under 2 seconds
    assert 0 <= result["score"] <= 1 and result["label"] in (0, 1)
    assert set(result["risk"]) == {"model", "rules", "overall"}
    assert result["risk"]["rules"] == "High"  # R1 (High) fired
    assert {"R1", "R3", "R4"} <= {r["rule_id"] for r in result["rules"]}
    assert len(result["shap"]) <= 10 and result["reasons"]
    assert "<mark" in result["highlight_html"] or result["highlight_html"]


def test_analyze_text_escapes_html(store) -> None:
    result = service.analyze(TEXT, "text", "LR", store)
    assert "<b>" not in result["highlight_html"] and "&lt;b&gt;" in result["highlight_html"]
    assert result["model_id"] == "lr_text"


def test_lime_compare_and_health(store) -> None:
    result = service.analyze(TEXT, "text", "LR", store)
    lime = service.lime_compare(TEXT, "text", "lr_text", result["words"], store, num_samples=200)
    assert 0 <= lime["jaccard_top5"] <= 1 and lime["lime"] and lime["seconds"] < 15
    assert service.health(store)["status"] == "ok"
    options = service.category_options(store)
    assert "Unknown" in options["employment_type"]
