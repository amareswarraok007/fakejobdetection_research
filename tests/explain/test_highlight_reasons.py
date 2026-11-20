"""Tests for cvx.explain.highlight (character spans, safe HTML) and cvx.explain.reasons."""

import pandas as pd

from cvx.explain.core import Explanation
from cvx.explain.highlight import Span, render_html, word_spans
from cvx.explain.reasons import CAUTION, format_reasons, reasons


def test_word_spans_case_insensitive_whole_words() -> None:
    text = "Pay a Fee now. Feeling fine? fee!"
    spans = word_spans(text, {"fee": 0.5, "now": -0.2, "urltoken": 1.0})
    assert [(s.start, s.end, s.word) for s in spans] == [
        (6, 9, "fee"),
        (10, 13, "now"),
        (29, 32, "fee"),
    ]
    assert text[6:9] == "Fee"  # 'Feeling' is not matched


def test_render_html_escapes_and_marks() -> None:
    text = "<script>x</script> fee"
    html = render_html(text, [Span(19, 22, "fee", 0.4)])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<sup>+</sup>" in html and 'title="fee: +0.400 (towards fake)"' in html
    genuine = render_html("team", [Span(0, 4, "team", -0.4)])
    assert "<sup>−</sup>" in genuine and "towards genuine" in genuine


def test_reasons_for_words_and_metadata() -> None:
    exp = Explanation(
        "shap",
        0.9,
        0.0,
        features={
            "registration fee": 0.8,
            "team": -0.3,
            "urltoken": 0.2,
            "numeric__has_company_logo": 0.5,
            "category__employment_type_Part-time": 0.4,
            "category__employment_type_Full-time": 0.6,
        },
    )
    row = pd.DataFrame([{"has_company_logo": 0.0, "employment_type": "Part-time"}])
    sentences = reasons(exp, row, top=5)
    assert sentences[0] == "The model links the phrase ‘registration fee’ with fake postings."
    assert "The posting has no company logo; this pushes the score towards fake." in sentences
    assert "Employment type is ‘Part-time’; this pushes the score towards fake." in sentences
    # 'Full-time' is not this posting's value, so it gives no reason.
    assert not any("Full-time" in s for s in sentences)
    assert "The model links the word ‘team’ with genuine postings." in sentences
    assert "The model links a web link in the text with fake postings." in sentences


def test_format_reasons_has_caution() -> None:
    assert format_reasons(["x"]) == {"reasons": ["x"], "caution": CAUTION}
