"""Tests for cvx.features.build: model input table and the ColumnTransformer."""

import pandas as pd
from scipy import sparse

from cvx.config import TfidfConfig
from cvx.features.build import TEXT_COLUMN, make_model_input, make_preprocessor
from cvx.features.meta import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TFIDF = TfidfConfig(ngram_range=[1, 1], min_df=1, max_df=1.0, max_features=None, sublinear_tf=False)


def test_text_only_input(tiny_postings: pd.DataFrame) -> None:
    for method in ("p_cv", "p_cvx"):
        table = make_model_input(tiny_postings, method, metadata=False)
        assert list(table.columns) == [TEXT_COLUMN]
        assert list(table.index) == list(tiny_postings.index)
    assert make_model_input(tiny_postings, "p_cvx", False)[TEXT_COLUMN][0].startswith("title: ")


def test_full_input_has_metadata(tiny_postings: pd.DataFrame) -> None:
    table = make_model_input(tiny_postings, "p_cvx", metadata=True)
    assert list(table.columns) == [TEXT_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def test_preprocessor_text_only(corpus: pd.DataFrame) -> None:
    table = make_model_input(corpus, "p_cvx", metadata=False)
    matrix = make_preprocessor(TFIDF, metadata=False).fit_transform(table)
    assert sparse.issparse(matrix) and matrix.shape[0] == len(corpus)


def test_preprocessor_full_is_sparse_and_non_negative(corpus: pd.DataFrame) -> None:
    table = make_model_input(corpus, "p_cvx", metadata=True)
    pre = make_preprocessor(TFIDF, metadata=True)
    matrix = pre.fit_transform(table)
    assert sparse.issparse(matrix)
    assert matrix.min() >= 0  # Naive Bayes needs non-negative features
    names = pre.get_feature_names_out()
    assert any(n.startswith("numeric__") for n in names)
    assert any(n.startswith("category__employment_type") for n in names)


def test_function_keeps_top_20_plus_other(make_postings) -> None:
    rows = [{"title": f"job {i}", "function": f"F{i:02d}"} for i in range(30)]
    table = make_model_input(make_postings(rows), "p_cvx", metadata=True)
    pre = make_preprocessor(TFIDF, metadata=True).fit(table)
    function_columns = [n for n in pre.get_feature_names_out() if n.startswith("function__")]
    assert len(function_columns) == 21  # 20 values + one "infrequent" (Other) bucket
    unseen = table.iloc[[0]].assign(function="Never Seen")
    assert pre.transform(unseen).shape[1] == len(pre.get_feature_names_out())
