"""Load and validate experiment configs (YAML files in ``configs/``).

A config may say ``extends: base.yaml``. The parent file is loaded first and the
child's values are merged on top (nested dictionaries merge key by key; lists and
plain values are replaced). The merged result is turned into frozen dataclasses.

Any key that is not defined below is rejected with a clear error. This catches
typos such as ``thresold`` that would otherwise be silently ignored.

An experiment can have *variants* (for example E1a/E1b/E1c). Each variant is the
experiment config with a few keys overridden; ``expand_variants`` builds one full,
validated ``Config`` per variant.
"""

import dataclasses
import re
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
TUNING_METHODS = ("validation", "grouped_cv")
# "train": E0-E4 (fit and evaluate models). The others work on exported models (E5-E7)
# or on the data alone (E8).
TASKS = ("train", "explain", "faithfulness", "audit", "rules")
AUDIT_COLUMNS = ("location", "industry", "department")
SCORING_METRICS = ("average_precision", "roc_auc")
RESAMPLE_METHODS = ("random_oversample", "smote")
RESAMPLE_WHEN = ("before_split", "train_only")
# Speed mode (PRD §9.1): at most this many values per hyperparameter.
MAX_GRID_VALUES = 3
DEFAULT_VARIANT = "main"
_VARIANT_NAME = re.compile(r"^[a-z0-9_]+$")
_NOT_OVERRIDABLE = ("experiment_id", "variants", "compare_to")


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
    # E7a shortcut audit only: extra category columns (location, industry, department).
    extra_categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TuningConfig:
    """How hyperparameters are chosen: on the validation split or by grouped CV."""

    method: str
    scoring: str
    cv_folds: int | None = None
    # Per-model method, e.g. {xgb: validation} (speed mode, PRD §9.1).
    overrides: dict[str, str] = field(default_factory=dict)

    def for_model(self, model_id: str) -> "TuningConfig":
        """The tuning settings for one model (its override, if any, replaces ``method``)."""
        if model_id not in self.overrides:
            return self
        return dataclasses.replace(self, method=self.overrides[model_id], overrides={})


@dataclass(frozen=True)
class ResampleConfig:
    """Balancing the classes: which method, and before the split or inside training only."""

    method: str
    when: str


@dataclass(frozen=True)
class StatsConfig:
    """Statistical tests on the test split.

    ``mcnemar`` is a list of pairs; each item is ``"variant/model"`` or just ``"model"``
    when the experiment has one variant.
    """

    bootstrap: int | None = None
    mcnemar: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class VariantConfig:
    """One named variant: the experiment config with ``overrides`` merged on top."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class XaiConfig:
    """Settings for the explanation experiments E5-E7 (PRD Section 8.2; speed mode §9.1)."""

    model_store: str = "app/model_store"
    models: list[str] = field(default_factory=list)
    n_postings: int = 100
    n_check: int = 20
    n_examples: int = 10
    n_global: int = 1000
    ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    random_draws: int = 5
    lime_samples: int = 1000
    lime_features: int = 20
    lime_seeds: int = 3
    top_global: int = 30
    n_errors: int = 20
    audit_columns: list[str] = field(default_factory=lambda: ["location", "industry"])
    mask_tokens: list[str] = field(default_factory=list)


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
    grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    tuning: TuningConfig | None = None
    thresholds: list[float | str] = field(default_factory=list)
    resample: ResampleConfig | None = None
    stats: StatsConfig | None = None
    variants: list[VariantConfig] = field(default_factory=list)
    compare_to: str | None = None
    export_models: bool = False
    assumptions: list[str] = field(default_factory=list)
    task: str = "train"
    xai: XaiConfig | None = None


def load_config(path: str | Path) -> Config:
    """Load a YAML config, follow ``extends``, reject unknown keys and validate values.

    Input: path to a YAML file. Output: a frozen ``Config``.
    Raises ``ConfigError`` with a readable message on any problem.
    """
    merged = _load_merged(Path(path), seen=())
    config = _build(Config, merged, where="config")
    _validate(config)
    for _, variant_config in expand_variants(config):
        _validate(variant_config)
    return config


def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a ``Config`` back to plain dictionaries (for saving with a run)."""
    return dataclasses.asdict(config)


def expand_variants(config: Config) -> list[tuple[str, Config]]:
    """Return ``(variant_name, config)`` pairs; one ``("main", config)`` if there are none.

    Each variant config is the experiment config with the variant's ``overrides``
    deep-merged on top, and with ``variants``/``compare_to`` removed.
    """
    base = config_to_dict(config)
    base.update(variants=[], compare_to=None)
    if not config.variants:
        return [(DEFAULT_VARIANT, _build(Config, base, where="config"))]
    expanded = []
    for variant in config.variants:
        where = f"config.variants.{variant.name}"
        merged = _deep_merge(base, variant.overrides)
        expanded.append((variant.name, _build(Config, merged, where=where)))
    return expanded


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


def _dataclass_in(annotation: Any) -> type | None:
    """If a type is a dataclass (or ``Dataclass | None``), return that dataclass."""
    if dataclasses.is_dataclass(annotation):
        return annotation
    if typing.get_origin(annotation) in (Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if dataclasses.is_dataclass(arg):
                return arg
    return None


def _convert(annotation: Any, value: Any, where: str) -> Any:
    """Build nested dataclasses (single or in a list) from plain YAML values."""
    if value is None:
        return None
    sub = _dataclass_in(annotation)
    if sub is not None:
        return _build(sub, value, where)
    if typing.get_origin(annotation) is list:
        (item_type,) = typing.get_args(annotation) or (Any,)
        item_class = _dataclass_in(item_type)
        if item_class is not None:
            if not isinstance(value, list):
                raise ConfigError(f"'{where}' must be a list.")
            return [_build(item_class, v, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


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
        kwargs[f.name] = _convert(hints[f.name], data[f.name], f"{where}.{f.name}")
    return cls(**kwargs)


# --- validation ------------------------------------------------------------------
def _validate(config: Config) -> None:
    """Check values that the type system cannot (allowed names, ratios, ranges)."""
    if not isinstance(config.seed, int):
        raise ConfigError(f"'seed' must be an integer, got {config.seed!r}.")
    _check_choice("preprocessing", config.preprocessing, PREPROCESSING_METHODS)
    _validate_split(config.split)
    unknown_models = [m for m in config.models if m not in MODEL_IDS]
    if unknown_models:
        raise ConfigError(f"Unknown model id(s) {unknown_models}. Allowed: {list(MODEL_IDS)}")
    for threshold in config.thresholds:
        _validate_threshold(threshold)
    _validate_grids(config.grids)
    if config.tuning is not None:
        _validate_tuning(config.tuning, config.split)
    _check_choice("task", config.task, TASKS)
    unknown_extra = [c for c in config.features.extra_categories if c not in AUDIT_COLUMNS]
    if unknown_extra:
        raise ConfigError(f"Unknown features.extra_categories {unknown_extra}.")
    if config.task in ("explain", "faithfulness", "audit"):
        _validate_xai(config.xai)
    if config.models and config.tuning is None:
        raise ConfigError("'tuning' is required when 'models' is set.")
    if config.resample is not None:
        _validate_resample(config)
    if config.stats is not None:
        _validate_stats(config.stats)
    _validate_variants(config)


def _check_choice(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    """Raise if ``value`` is not one of ``allowed``."""
    if value not in allowed:
        raise ConfigError(f"'{name}' must be one of {allowed}, got {value!r}.")


def _validate_split(split: SplitConfig) -> None:
    """Check the split method, the three ratios and grouped-split settings."""
    _check_choice("split.method", split.method, SPLIT_METHODS)
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


def _validate_grids(grids: dict[str, dict[str, list[Any]]]) -> None:
    """Grid model ids must be known; each hyperparameter has 1 to 3 values (speed mode)."""
    for model_id, grid in grids.items():
        _check_choice("grids key", model_id, MODEL_IDS)
        if not isinstance(grid, dict):
            raise ConfigError(f"'grids.{model_id}' must map parameter names to lists.")
        for name, values in grid.items():
            if not isinstance(values, list) or not 1 <= len(values) <= MAX_GRID_VALUES:
                raise ConfigError(
                    f"'grids.{model_id}.{name}' must be a list of 1 to {MAX_GRID_VALUES} values "
                    "(speed mode, PRD §9.1)."
                )


def _validate_tuning(tuning: TuningConfig, split: SplitConfig) -> None:
    """Grouped CV needs a fold count and a grouped split (groups come from the split)."""
    _check_choice("tuning.method", tuning.method, TUNING_METHODS)
    _check_choice("tuning.scoring", tuning.scoring, SCORING_METRICS)
    for model_id, method in tuning.overrides.items():
        _check_choice("tuning.overrides key", model_id, MODEL_IDS)
        _check_choice(f"tuning.overrides.{model_id}", method, TUNING_METHODS)
    if tuning.method == "grouped_cv":
        if tuning.cv_folds is None or tuning.cv_folds < 2:
            raise ConfigError("'tuning.cv_folds' must be an integer >= 2 for grouped_cv.")
        if split.method != "grouped_stratified":
            raise ConfigError(
                "'tuning.method: grouped_cv' needs 'split.method: grouped_stratified'."
            )


def _validate_resample(config: Config) -> None:
    """Allowed combinations of resampling method, timing, split and models."""
    resample = config.resample
    assert resample is not None
    _check_choice("resample.method", resample.method, RESAMPLE_METHODS)
    _check_choice("resample.when", resample.when, RESAMPLE_WHEN)
    if resample.when == "before_split":
        if config.split.method != "random_stratified":
            raise ConfigError(
                "Resampling before the split is only defined for a random split (E1)."
            )
        if resample.method == "smote" and config.features.metadata:
            raise ConfigError("SMOTE before the split works on TF-IDF text features only.")
    unsupported = [m for m in config.models if m in ("xgb", "linear_svm")]
    if resample.when == "train_only" and unsupported:
        # xgb needs its own fit path (early stopping) and linear_svm's grouped calibration
        # folds would point at the wrong rows once the sampler adds rows.
        raise ConfigError(f"Resampling inside training is not supported for {unsupported}.")


def _validate_stats(stats: StatsConfig) -> None:
    """Bootstrap count is positive; every McNemar entry is a pair."""
    if stats.bootstrap is not None and (
        not isinstance(stats.bootstrap, int) or stats.bootstrap < 1
    ):
        raise ConfigError("'stats.bootstrap' must be a positive integer.")
    for pair in stats.mcnemar:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError(f"Each 'stats.mcnemar' entry must be a pair, got {pair!r}.")


def _validate_variants(config: Config) -> None:
    """Variant names are unique and simple; overrides may not change the experiment itself."""
    names = [v.name for v in config.variants]
    if len(set(names)) != len(names):
        raise ConfigError(f"Variant names must be unique, got {names}.")
    for variant in config.variants:
        if not _VARIANT_NAME.match(variant.name):
            raise ConfigError(f"Variant name {variant.name!r} may use only a-z, 0-9 and '_'.")
        blocked = [k for k in variant.overrides if k in _NOT_OVERRIDABLE]
        if blocked:
            raise ConfigError(f"Variant {variant.name!r} may not override {blocked}.")
    if config.compare_to is not None and config.compare_to not in names:
        raise ConfigError(f"'compare_to' must name a variant, got {config.compare_to!r}.")


def _validate_xai(xai: XaiConfig | None) -> None:
    """E5-E7 need an ``xai`` section naming exported models; sizes must be positive."""
    if xai is None or not xai.models:
        raise ConfigError(
            "Tasks explain/faithfulness/audit need 'xai.models' (exported model ids)."
        )
    sizes = (xai.n_postings, xai.n_check, xai.lime_samples, xai.lime_seeds, xai.random_draws)
    if any(not isinstance(v, int) or v < 1 for v in sizes) or not xai.ks:
        raise ConfigError(
            "xai sizes (n_postings, n_check, lime_samples, …) must be positive integers."
        )
    unknown = [c for c in xai.audit_columns if c not in AUDIT_COLUMNS]
    if unknown:
        raise ConfigError(f"Unknown xai.audit_columns {unknown}. Allowed: {list(AUDIT_COLUMNS)}")
