"""Tests for cvx.audit.shortcuts and cvx.audit.errors on hand examples."""

import numpy as np
import pandas as pd

from cvx.audit.errors import confident_errors
from cvx.audit.shortcuts import MASK, audit_share, flag_tokens, mask_text, place_words
from cvx.xai_eval.experiment import select_sample


def test_place_words() -> None:
    postings = pd.DataFrame({"location": ["US, TX, Houston", "GB, LND, London", None, "  "]})
    assert place_words(postings, frozenset({"london"})) == {"lnd", "houston"}


def test_flag_and_mask() -> None:
    ranking = [
        {"feature": "text__houston office"},
        {"feature": "text__fee"},
        {"feature": "numeric__has_company_logo"},
    ]
    flagged = flag_tokens(ranking, {"houston", "dallas"}, ["Acme"])
    assert flagged == ["houston", "acme"]
    masked = mask_text(pd.Series(["houston houstonian acme fee"]), flagged)
    assert masked[0] == f"{MASK} houstonian {MASK} fee"


def test_audit_share() -> None:
    ranking = [{"feature": "audit__location_US"}, {"feature": "text__fee"}]
    assert audit_share(ranking) == 0.5 and audit_share([]) == 0.0


def test_confident_errors() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.1, 0.3, 0.9, 0.8, 0.95, 0.2])
    picked = confident_errors(y, scores, threshold=0.5, n=1)
    assert picked["false_negative"].tolist() == [0]  # fake with the lowest score
    assert picked["false_positive"].tolist() == [4]  # genuine with the highest score


def test_select_sample_is_balanced_and_seeded() -> None:
    y = np.array([1] * 10 + [0] * 10)
    predicted = np.array([1] * 8 + [0] * 2 + [0] * 9 + [1])
    sample = select_sample(y, predicted, n=6, seed=42)
    assert len(sample) == 6 and np.array_equal(sample, select_sample(y, predicted, 6, 42))
    assert (y[sample] == predicted[sample]).all()  # only true positives / true negatives
