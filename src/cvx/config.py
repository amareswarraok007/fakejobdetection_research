"""Load and validate experiment configs (YAML files in ``configs/``).

A config may say ``extends: base.yaml``. The parent file is loaded first and the
child's values are merged on top (nested dictionaries merge key by key; lists and
plain values are replaced). The merged result is turned into frozen dataclasses.

Any key that is not defined below is rejected with a clear error. This catches
typos such as ``thresold`` that would otherwise be silently ignored. Later
milestones add new keys here when their experiments need them.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

PREPROCESSING_METHODS = ("p_cv", "p_cvx")
SPLIT_METHODS = ("random_stratified", "grouped_stratified")
MODEL_IDS = ("nb", "linear_svm", "rf", "lr", "xgb")
NAMED_THRESHOLDS = ("t_f1", "t_r90")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or contains unknown keys."""


@dataclass(frozen=True)
class PathsConfig:
    """File locations, relative to the repository root."""

    data_csv: str
    runs_dir: str
    reports_dir: str


@dataclass(frozen=True)
class SplitConfig:
    """How postings are divided into train / validation / test."""

    method: str
    ratios: list[float]
    near_dup_threshold: float | None = None
    n_folds: int | None = None


@dataclass(frozen=True)
class TfidfConfig:
    """Settings passed to scikit-learn's TfidfVectorizer."""

    ngram_range: list[int]
    min_df: int | float
    max_df: int | float
    max_features: int | None
    sublinear_tf: bool


@dataclass(frozen=True)
class FeaturesConfig:
    """Which feature groups the model uses."""

    text: bool = True
    metadata: bool = False


@dataclass(frozen=True)
class TuningConfig:
    """Hyper-parameter tuning by grouped cross-validation."""

    cv_folds: int
    scoring: str


@dataclass(frozen=True)
class StatsConfig:
    """Statistical tests run on the test split."""

    bootstrap: int | None = None
    mcnemar: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    """One fully merged and validated experiment config."""

    experiment_id: str
    seed: int
    paths: PathsConfig
    preprocessing: str
    split: SplitConfig
    tfidf: TfidfConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    models: list[str] = field(default_factory=list)
    tuning: TuningConfig | None = None
    thresholds: list[float | str] = field(default_factory=list)
    stats: StatsConfig | None = None


def load_config(path: str | Path) -> Config:
    """Load a YAML config, follow ``extends``, reject unknown keys and validate values.

    Input: path to a YAML file. Output: a frozen ``Config``.
    Raises ``ConfigError`` with a readable message on any problem.
    """
    merged = _load_merged(Path(path), seen=())
    config = _build(Config, merged, where="config")
    _validate(config)
    return config


def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a ``Config`` back to plain dictionaries (for saving with a run)."""
    return dataclasses.asdict(config)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file and check that its top level is a mapping."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping of keys to values.")
    return data


def _load_merged(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    """Read ``path`` and recursively merge it on top of the file it extends."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"Circular 'extends' chain: {chain}")
    data = _read_yaml(path)
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    # The parent path is relative to the folder of the child file.
    parent = _load_merged(path.parent / str(parent_name), seen=(*seen, resolved))
    return _deep_merge(parent, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` on top of ``base``; nested dicts merge key by key."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _nested_dataclass(annotation: Any) -> type | None:
    """If a field's type is a dataclass (or ``Dataclass | None``), return that dataclass."""
    if dataclasses.is_dataclass(annotation):
        return annotation
    if typing.get_origin(annotation) in (Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if dataclasses.is_dataclass(arg):
                return arg
    return None


def _build(cls: type, data: Any, where: str) -> Any:
    """Create dataclass ``cls`` from ``data``; reject unknown and missing keys.

    ``where`` is the dotted location used in error messages (e.g. ``config.split``).
    """
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}.")
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where}': {unknown}. Allowed: {sorted(allowed)}")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            no_default = f.default is dataclasses.MISSING
            if no_default and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"Missing required key '{where}.{f.name}'.")
            continue
        value = data[f.name]
        sub = _nested_dataclass(hints[f.name])
        if sub is not None and value is not None:
            value = _build(sub, value, where=f"{where}.{f.name}")
        kwargs[f.name] = value
    return cls(**kwargs)


def _validate(config: Config) -> None:
    """Check values that the type system cannot (allowed names, ratios, ranges)."""
    if not isinstance(config.seed, int):
        raise ConfigError(f"'seed' must be an integer, got {config.seed!r}.")
    if config.preprocessing not in PREPROCESSING_METHODS:
        raise ConfigError(
            f"'preprocessing' must be one of {PREPROCESSING_METHODS}, got {config.preprocessing!r}."
        )
    _validate_split(config.split)
    unknown_models = [m for m in config.models if m not in MODEL_IDS]
    if unknown_models:
        raise ConfigError(f"Unknown model id(s) {unknown_models}. Allowed: {list(MODEL_IDS)}")
    for threshold in config.thresholds:
        _validate_threshold(threshold)


def _validate_split(split: SplitConfig) -> None:
    """Check the split method, the three ratios and grouped-split settings."""
    if split.method not in SPLIT_METHODS:
        raise ConfigError(f"'split.method' must be one of {SPLIT_METHODS}, got {split.method!r}.")
    if len(split.ratios) != 3 or any(r <= 0 for r in split.ratios):
        raise ConfigError("'split.ratios' must be three positive numbers (train, val, test).")
    if abs(sum(split.ratios) - 1.0) > 1e-9:
        raise ConfigError(f"'split.ratios' must sum to 1, got {split.ratios}.")
    if split.method == "grouped_stratified":
        threshold = split.near_dup_threshold
        if threshold is None or not 0 < threshold <= 1:
            raise ConfigError("'split.near_dup_threshold' must be in (0, 1] for a grouped split.")
        if split.n_folds is None or split.n_folds < 3:
            raise ConfigError("'split.n_folds' must be an integer >= 3 for a grouped split.")


def _validate_threshold(threshold: float | str) -> None:
    """A threshold is either a number in [0, 1] or a named one (t_f1, t_r90)."""
    if isinstance(threshold, str):
        if threshold not in NAMED_THRESHOLDS:
            raise ConfigError(
                f"Unknown threshold {threshold!r}. Use a number or one of {NAMED_THRESHOLDS}."
            )
    elif isinstance(threshold, bool) or not 0 <= float(threshold) <= 1:
        raise ConfigError(f"Threshold {threshold!r} must be between 0 and 1.")
