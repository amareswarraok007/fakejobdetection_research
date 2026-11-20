"""PRD M4 checks: linear SHAP = shap.LinearExplainer (1e-6); Tree SHAP additivity (1e-4)."""

import re
from typing import Any

import numpy as np
import pytest

from cvx.explain.dispatch import explain_shap, model_kind, shap_matrix
from cvx.explain.shap_linear import check_against_shap, linear_shap
from cvx.explain.shap_tree import additivity_error


@pytest.mark.parametrize("name", ["lr_text", "lr_full"])
def test_linear_formula_matches_shap_on_20_postings(
    fitted_models: dict[str, Any], name: str
) -> None:
    rows = fitted_models["table"].iloc[:20]
    model, means = fitted_models["models"][name], fitted_models["means"][name]
    assert check_against_shap(model, rows, means) < 1e-6


def test_linear_values_add_up_to_log_odds(fitted_models: dict[str, Any]) -> None:
    model, means = fitted_models["models"]["lr_full"], fitted_models["means"]["lr_full"]
    rows = fitted_models["table"].iloc[:20]
    phi, base, _ = linear_shap(model, rows, means)
    assert np.allclose(base + phi.sum(axis=1), model.decision_function(rows), atol=1e-9)


@pytest.mark.parametrize("name", ["xgb_text", "xgb_full"])
def test_tree_shap_additivity(fitted_models: dict[str, Any], name: str) -> None:
    rows = fitted_models["table"].iloc[:20]
    assert additivity_error(fitted_models["models"][name], rows) < 1e-4


@pytest.mark.parametrize("name", ["lr_full", "xgb_full"])
def test_explanations_cover_present_words(fitted_models: dict[str, Any], name: str) -> None:
    table = fitted_models["table"]
    model, means = fitted_models["models"][name], fitted_models["means"][name]
    exp = explain_shap(model, table.iloc[[0]], means)[0]
    assert 0 <= exp.score <= 1 and exp.method == "shap"
    posting_words = set(re.findall(r"\w+", table["text"].iloc[0]))
    assert set(exp.words) <= posting_words  # only words that are in the posting
    assert any(name.startswith("numeric__") for name in exp.features)
    # All SHAP mass is accounted for: present features + absent words.
    total = sum(exp.features.values()) + exp.absent_total
    assert total == pytest.approx(shap_matrix(model, table.iloc[[0]], means).sum(), abs=1e-6)


def test_model_kind(fitted_models: dict[str, Any]) -> None:
    assert model_kind(fitted_models["models"]["lr_text"]) == "linear"
    assert model_kind(fitted_models["models"]["xgb_text"]) == "tree"
    with pytest.raises(ValueError, match="means"):
        explain_shap(fitted_models["models"]["lr_text"], fitted_models["table"].iloc[[0]])
