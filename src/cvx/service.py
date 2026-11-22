"""The only functions the Streamlit app calls (PRD §11.6). No Streamlit import here,
so everything can be tested with pytest.

- ``open_store``: load and check the model store (hash, versions, smoke scores).
- ``analyze``: score, label, thresholds, SHAP list, highlights, reasons, rules, risk.
- ``lime_compare``: LIME top-10 words and their top-5 Jaccard overlap with SHAP.
- ``health``: loaded models, versions and the smoke-check result.
"""

from dataclasses import asdict
from typing import Any

from cvx.artifacts.load import ModelStore, load_store
from cvx.explain.core import Explanation, top_words
from cvx.explain.dispatch import explain_shap
from cvx.explain.highlight import render_html, word_spans
from cvx.explain.lime_text import explain_lime
from cvx.explain.reasons import CAUTION, reasons
from cvx.rules.safety import check_all, model_level, overall_level, rules_level
from cvx.schema import build_input, display_text, rules_text
from cvx.xai_eval.agreement import jaccard

CHOICES = ("Auto", "LR", "XGBoost")
_PREFIX = {"LR": "lr", "XGBoost": "xgb"}
TOP_FEATURES = 10
LIME_SAMPLES = 1000


def available_choices(store: ModelStore, mode: str) -> tuple[str, ...]:
    """Dropdown choices supported by the loaded model store for ``mode``."""
    suffix = "full" if mode == "full" else "text"
    present = [label for label, prefix in _PREFIX.items() if f"{prefix}_{suffix}" in store.models]
    return ("Auto", *present) if present else ("Auto",)


def open_store(path: str) -> ModelStore:
    """Load the model store; raises ``ModelStoreError`` if any safety check fails."""
    return load_store(path)


def choose_model(store: ModelStore, mode: str, choice: str) -> str:
    """Model id for the mode ('full' -> *_full, 'text' -> *_text) and the user's choice.

    'Auto' picks the available model with the higher test PR-AUC in the model card.
    """
    suffix = "full" if mode == "full" else "text"
    if choice in _PREFIX:
        model_id = f"{_PREFIX[choice]}_{suffix}"
        if model_id not in store.models:
            raise KeyError(f"Model {model_id} is not in the model store.")
        return model_id
    options = [m for m in store.models if m.endswith(f"_{suffix}")]
    if not options:
        raise KeyError(f"No *_{suffix} model in the model store.")

    def pr_auc(model_id: str) -> float:
        """Test PR-AUC copied from the training run into the card."""
        value = store.models[model_id].info["test_metrics"]["threshold_free"]["pr_auc"]
        return float(value or 0.0)

    return max(sorted(options), key=pr_auc)


def _shap_list(explanation: Explanation) -> list[dict[str, Any]]:
    """Top features of the posting by absolute SHAP value (for the chart and table)."""
    items = sorted(explanation.features.items(), key=lambda item: (-abs(item[1]), item[0]))
    return [{"feature": name, "shap": value} for name, value in items[:TOP_FEATURES]]


def analyze(form: dict[str, Any], mode: str, choice: str, store: ModelStore) -> dict[str, Any]:
    """Full analysis of one posting. Raises ``InputError`` for an incomplete form."""
    row = build_input(form, mode)
    model_id = choose_model(store, mode, choice)
    loaded = store.models[model_id]
    explanation = explain_shap(loaded.estimator, row, loaded.means)[0]
    thresholds = loaded.info["thresholds"]
    hits = check_all(
        rules_text(form, mode), form.get("recruiter_email"), form.get("company_website")
    )
    levels = {
        "model": model_level(explanation.score, thresholds["t_f1"], thresholds["t_r90"]),
        "rules": rules_level(hits),
    }
    levels["overall"] = overall_level(levels["model"], levels["rules"])
    text = display_text(form, mode)
    return {
        "model_id": model_id,
        "score": explanation.score,
        "label": int(explanation.score >= thresholds["t_f1"]),
        "thresholds": thresholds,
        "shap": _shap_list(explanation),
        "words": explanation.words,
        "highlight_html": render_html(text, word_spans(text, explanation.words)),
        "reasons": reasons(explanation, row),
        "caution": CAUTION,
        "rules": [asdict(hit) for hit in hits],
        "risk": levels,
    }


def lime_compare(
    form: dict[str, Any],
    mode: str,
    model_id: str,
    shap_words: dict[str, float],
    store: ModelStore,
    num_samples: int = LIME_SAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """LIME top-10 words for the same posting and model, and top-5 Jaccard with SHAP."""
    row = build_input(form, mode)
    lime, seconds = explain_lime(store.models[model_id].estimator, row, num_samples, 10, seed)
    shap = Explanation("shap", lime.score, None, words=shap_words)
    return {
        "lime": [{"word": w, "weight": lime.words[w]} for w in top_words(lime, 10)],
        "jaccard_top5": jaccard(top_words(shap, 5), top_words(lime, 5)),
        "seconds": seconds,
    }


def health(store: ModelStore) -> dict[str, Any]:
    """Loaded models, library versions and the smoke-check result."""
    return dict(store.report)


def category_options(store: ModelStore) -> dict[str, list[str]]:
    """Dropdown choices learned by a full model's encoders (no hand-typed lists)."""
    full = next((m for m in store.models.values() if m.info.get("metadata")), None)
    if full is None:
        return {}
    features = full.estimator.named_steps["features"]
    options: dict[str, list[str]] = {}
    for name in ("category", "function"):
        encoder = features.named_transformers_[name]
        for column, values in zip(encoder.feature_names_in_, encoder.categories_, strict=True):
            options[str(column)] = sorted({str(v) for v in values} | {"Unknown"})
    return options
