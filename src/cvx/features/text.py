"""TF-IDF text features.

TF-IDF gives each word a score that is high when the word is frequent in this
posting but rare across all postings. Settings come from the config's ``tfidf``
section (PRD Section 6.4): scikit-learn defaults for E0, our settings for P-CVX.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from cvx.config import TfidfConfig


def make_tfidf(settings: TfidfConfig) -> TfidfVectorizer:
    """Create an unfitted ``TfidfVectorizer`` from the config settings.

    Input: ``TfidfConfig``. Output: ``TfidfVectorizer`` (fit it on training text only).
    """
    low, high = settings.ngram_range
    return TfidfVectorizer(
        ngram_range=(int(low), int(high)),
        min_df=settings.min_df,
        max_df=settings.max_df,
        max_features=settings.max_features,
        sublinear_tf=settings.sublinear_tf,
    )
