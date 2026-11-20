"""Tests for cvx.features.text: config settings reach the TfidfVectorizer."""

from cvx.config import TfidfConfig
from cvx.features.text import make_tfidf


def test_make_tfidf_uses_settings() -> None:
    settings = TfidfConfig(
        ngram_range=[1, 2], min_df=1, max_df=0.9, max_features=50, sublinear_tf=True
    )
    vectorizer = make_tfidf(settings)
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.max_df == 0.9 and vectorizer.max_features == 50
    assert vectorizer.sublinear_tf is True


def test_make_tfidf_bigrams_on_text() -> None:
    settings = TfidfConfig(
        ngram_range=[1, 2], min_df=1, max_df=1.0, max_features=None, sublinear_tf=False
    )
    vectorizer = make_tfidf(settings).fit(["no experience needed", "experience required"])
    vocabulary = set(vectorizer.get_feature_names_out())
    assert {"no experience", "experience needed", "no"} <= vocabulary
