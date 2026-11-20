"""Tuning, final fit and threshold choice (PRD Sections 6.5 and 8.2).

Order of work for one model (the test split is never touched here):
1. Tune: try every grid candidate, either on the validation split (E0/E1) or by
   grouped k-fold CV inside the training split (E2-E4). Score = PR-AUC by default.
2. Fit the best candidate on the training split only.
3. Choose thresholds on the validation split:
   - ``t_f1``: the threshold with the best fake-class F1;
   - ``t_r90``: the HIGHEST threshold whose validation recall is still >= 0.90.

Speed: the features (TF-IDF, metadata) never depend on the model or its
hyperparameters, so they are fitted ONCE per fold (``FoldData``) on that fold's
training rows and reused by every model and candidate. Results are identical to
refitting them each time; only the time changes.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from cvx.config import Config, TuningConfig
from cvx.data.prepare import ModelData, take
from cvx.data.resample import make_sampler
from cvx.features.build import make_preprocessor
from cvx.models.zoo import make_classifier

logger = logging.getLogger(__name__)

RECALL_TARGET = 0.90
SCORERS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
}
# (params, y_fit, groups_fit) -> unfitted "head": the classifier, or sampler + classifier
HeadFactory = Callable[[dict[str, Any], np.ndarray, np.ndarray | None], Any]


@dataclass
class FoldData:
    """Feature matrices for one fit/evaluate pair. ``features`` was fitted on the fit rows."""

    features: Any | None
    X_fit: Any
    y_fit: np.ndarray
    groups_fit: np.ndarray | None
    X_eval: Any
    y_eval: np.ndarray


@dataclass
class TrainedModel:
    """A fitted model (features + classifier) with its tuning record and thresholds."""

    model_id: str
    estimator: Any
    best_params: dict[str, Any]
    best_score: float | None
    cv_rows: list[dict[str, Any]] = field(default_factory=list)
    val_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    thresholds: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def make_fold(
    data: ModelData, config: Config, fit_rows: np.ndarray, eval_rows: np.ndarray
) -> FoldData:
    """Fit the feature transformer on ``fit_rows`` only and transform both parts."""
    X_fit, X_eval = take(data.X, fit_rows), take(data.X, eval_rows)
    features = None
    if not data.prevectorized:
        features = make_preprocessor(
            config.tfidf, config.features.metadata, tuple(config.features.extra_categories)
        )
        X_fit, X_eval = features.fit_transform(X_fit), features.transform(X_eval)
    groups = None if data.groups is None else data.groups[fit_rows]
    return FoldData(features, X_fit, data.y[fit_rows], groups, X_eval, data.y[eval_rows])


def train_val_fold(data: ModelData, config: Config) -> FoldData:
    """Features fitted on the training split, evaluated on the validation split."""
    return make_fold(data, config, data.split.train, data.split.val)


def cv_folds(data: ModelData, config: Config) -> list[FoldData]:
    """Grouped stratified k folds inside the training split (same folds for every model)."""
    if data.groups is None:
        raise ValueError("Grouped CV needs near-duplicate groups (use a grouped split).")
    train = data.split.train
    splitter = StratifiedGroupKFold(
        n_splits=config.tuning.cv_folds, shuffle=True, random_state=config.seed
    )
    parts = splitter.split(np.zeros(len(train)), data.y[train], data.groups[train])
    return [make_fold(data, config, train[a], train[b]) for a, b in parts]


def make_head_factory(model_id: str, config: Config) -> HeadFactory:
    """Return a function building the classifier (with a sampler in front if configured)."""

    def head(params: dict[str, Any], y_fit: np.ndarray, groups: np.ndarray | None) -> Any:
        """Unfitted classifier, or imbalanced-learn Pipeline(sampler, classifier)."""
        classifier = make_classifier(model_id, params, y_fit, config.seed, groups)
        resample = config.resample
        if resample is not None and resample.when == "train_only":
            # imbalanced-learn applies the sampler during fit only, never at predict.
            sampler = make_sampler(resample.method, config.seed)
            return ImbPipeline([("resample", sampler), ("clf", classifier)])
        return classifier

    return head


def fit_head(head: Any, fold: FoldData) -> Any:
    """Fit on the fold's training part; XGBoost early-stops on the evaluation part."""
    if isinstance(head, XGBClassifier):
        return head.fit(
            fold.X_fit, fold.y_fit, eval_set=[(fold.X_eval, fold.y_eval)], verbose=False
        )
    return head.fit(fold.X_fit, fold.y_fit)


def predict_scores(estimator: Any, X: Any) -> np.ndarray:
    """Probability of the fake class for every row."""
    return np.asarray(estimator.predict_proba(X))[:, 1]


def assemble(features: Any | None, head: Any) -> Pipeline:
    """Join fitted features and a fitted head into one pipeline that predicts raw input."""
    steps: list[tuple[str, Any]] = [] if features is None else [("features", features)]
    if isinstance(head, ImbPipeline):
        return ImbPipeline(steps + list(head.steps))
    return Pipeline(steps + [("clf", head)])


def fit_and_score(
    make_head: HeadFactory, params: dict[str, Any], fold: FoldData, scoring: str
) -> tuple[Any, np.ndarray, float]:
    """Fit one candidate on a fold and score its evaluation part."""
    head = fit_head(make_head(params, fold.y_fit, fold.groups_fit), fold)
    scores = predict_scores(head, fold.X_eval)
    return head, scores, float(SCORERS[scoring](fold.y_eval, scores))


def select_best(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """Candidate with the highest mean score over its folds (first one wins a tie)."""
    means: dict[str, list[float]] = {}
    for row in rows:
        means.setdefault(json.dumps(row["params"], sort_keys=True), []).append(row["score"])
    best_key = max(means, key=lambda key: np.mean(means[key]))
    return json.loads(best_key), float(np.mean(means[best_key]))


def choose_thresholds(y_val: np.ndarray, val_scores: np.ndarray) -> dict[str, float]:
    """``t_f1`` (best F1) and ``t_r90`` (highest threshold with recall >= 0.90) on validation."""
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    return {
        "t_f1": float(thresholds[int(np.argmax(f1))]),
        "t_r90": float(thresholds[recall >= RECALL_TARGET].max()),
    }


def resolve_thresholds(requested: list[float | str], chosen: dict[str, float]) -> dict[str, float]:
    """Map the config's threshold list (e.g. [0.5, "t_f1"]) to ``{name: value}``."""
    return {str(t): (chosen[t] if isinstance(t, str) else float(t)) for t in requested}


def _tune(
    make_head: HeadFactory,
    candidates: list[dict[str, Any]],
    folds: list[FoldData] | None,
    train_val: FoldData,
    tuning: TuningConfig,
) -> tuple[dict[str, Any], float | None, list[dict[str, Any]], Any, np.ndarray]:
    """Pick the best candidate; return it with the head fitted on train and its val scores."""
    rows: list[dict[str, Any]] = []
    if tuning.method == "validation":
        fitted = []
        for params in candidates:
            head, scores, value = fit_and_score(make_head, params, train_val, tuning.scoring)
            rows.append({"params": params, "fold": "val", "score": value})
            fitted.append((head, scores))
        best, best_score = select_best(rows)
        head, scores = fitted[candidates.index(best)]  # no need to fit the winner twice
        return best, best_score, rows, head, scores
    best, best_score = candidates[0], None
    if len(candidates) > 1:  # nothing to choose for a fixed-settings model (rf)
        for k, fold in enumerate(folds or []):
            for params in candidates:
                _, _, value = fit_and_score(make_head, params, fold, tuning.scoring)
                rows.append({"params": params, "fold": k, "score": value})
        best, best_score = select_best(rows)
    head, scores, _ = fit_and_score(make_head, best, train_val, tuning.scoring)
    return best, best_score, rows, head, scores


def train_model(
    model_id: str,
    make_head: HeadFactory,
    candidates: list[dict[str, Any]],
    folds: list[FoldData] | None,
    train_val: FoldData,
    tuning: TuningConfig,
) -> TrainedModel:
    """Tune, fit on the training split, and choose thresholds on validation.

    ``folds`` (grouped CV) and ``train_val`` hold features that are already fitted.
    Output: ``TrainedModel`` whose ``estimator`` predicts from the raw model input.
    """
    start = time.perf_counter()
    best, best_score, rows, head, val_scores = _tune(
        make_head, candidates, folds, train_val, tuning
    )
    seconds = time.perf_counter() - start
    logger.info("%s: best %s (%.1f s)", model_id, best, seconds)
    return TrainedModel(
        model_id=model_id,
        estimator=assemble(train_val.features, head),
        best_params=best,
        best_score=best_score,
        cv_rows=rows,
        val_scores=val_scores,
        thresholds=choose_thresholds(train_val.y_eval, val_scores),
        seconds=seconds,
    )
