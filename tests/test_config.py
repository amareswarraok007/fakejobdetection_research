"""Tests for cvx.config: extends-merging, unknown-key rejection and value checks."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from cvx.config import ConfigError, config_to_dict, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID: dict[str, Any] = {
    "experiment_id": "parent",
    "seed": 42,
    "paths": {"data_csv": "data/x.csv", "runs_dir": "runs", "reports_dir": "reports"},
    "preprocessing": "p_cvx",
    "split": {
        "method": "grouped_stratified",
        "ratios": [0.7, 0.15, 0.15],
        "near_dup_threshold": 0.95,
        "n_folds": 20,
    },
    "tfidf": {
        "ngram_range": [1, 2],
        "min_df": 3,
        "max_df": 0.9,
        "max_features": 20000,
        "sublinear_tf": True,
    },
}


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as YAML to ``path`` and return the path."""
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_yaml_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert config.seed == 42
    assert config.preprocessing == "p_cvx"
    assert config.split.method == "grouped_stratified"
    assert config.split.ratios == [0.70, 0.15, 0.15]
    assert config.split.near_dup_threshold == 0.95
    assert config.features.metadata is False


def test_extends_merges_nested_keys(tmp_path: Path) -> None:
    write_yaml(tmp_path / "parent.yaml", VALID)
    child = {
        "extends": "parent.yaml",
        "experiment_id": "child",
        "preprocessing": "p_cv",
        "split": {"method": "random_stratified", "ratios": [0.8, 0.1, 0.1]},
        "models": ["lr"],
        "tuning": {"method": "validation", "scoring": "average_precision"},
    }
    config = load_config(write_yaml(tmp_path / "child.yaml", child))
    assert config.experiment_id == "child"
    assert config.preprocessing == "p_cv"
    assert config.split.method == "random_stratified"
    assert config.split.ratios == [0.8, 0.1, 0.1]
    # Not mentioned by the child, so inherited from the parent:
    assert config.split.near_dup_threshold == 0.95
    assert config.tfidf.max_features == 20000
    assert config.models == ["lr"]


def test_config_to_dict_round_trip(tmp_path: Path) -> None:
    config = load_config(write_yaml(tmp_path / "c.yaml", VALID))
    as_dict = config_to_dict(config)
    assert as_dict["split"]["n_folds"] == 20
    assert "extends" not in as_dict


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "c.yaml", {**VALID, "sead": 1})
    with pytest.raises(ConfigError, match="sead"):
        load_config(path)


def test_unknown_nested_key_rejected(tmp_path: Path) -> None:
    bad = {**VALID, "split": {**VALID["split"], "thresold": 0.9}}
    with pytest.raises(ConfigError, match=r"config\.split.*thresold"):
        load_config(write_yaml(tmp_path / "c.yaml", bad))


def test_missing_required_key_rejected(tmp_path: Path) -> None:
    bad = {k: v for k, v in VALID.items() if k != "seed"}
    with pytest.raises(ConfigError, match="seed"):
        load_config(write_yaml(tmp_path / "c.yaml", bad))


@pytest.mark.parametrize(
    "change",
    [
        {"preprocessing": "p_other"},
        {"split": {**VALID["split"], "ratios": [0.7, 0.2, 0.2]}},
        {"split": {**VALID["split"], "method": "random"}},
        {"split": {**VALID["split"], "near_dup_threshold": None}},
        {"models": ["lr", "bert"]},
        {"thresholds": [0.5, "t_best"]},
        {"thresholds": [1.5]},
    ],
)
def test_invalid_values_rejected(tmp_path: Path, change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path / "c.yaml", {**VALID, **change}))


def test_valid_thresholds_accepted(tmp_path: Path) -> None:
    good = {**VALID, "thresholds": [0.5, "t_f1", "t_r90"]}
    assert load_config(write_yaml(tmp_path / "c.yaml", good)).thresholds == [0.5, "t_f1", "t_r90"]


def test_circular_extends_rejected(tmp_path: Path) -> None:
    write_yaml(tmp_path / "a.yaml", {"extends": "b.yaml"})
    write_yaml(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------- M2/M3 keys
from cvx.config import expand_variants  # noqa: E402

MODELLED = {
    **VALID,
    "models": ["lr"],
    "tuning": {"method": "grouped_cv", "cv_folds": 5, "scoring": "average_precision"},
}


@pytest.mark.parametrize(
    "name", ["e0_reproduce", "e1_leakage", "e2_protocol", "e3_models", "e4_ablations"]
)
def test_repo_experiment_configs_load(name: str) -> None:
    config = load_config(REPO_ROOT / "configs" / f"{name}.yaml")
    assert config.experiment_id == name and config.seed == 42
    assert all(len(v) <= 3 for grid in config.grids.values() for v in grid.values())


def test_grid_with_more_than_three_values_rejected(tmp_path: Path) -> None:
    bad = {**MODELLED, "grids": {"lr": {"C": [0.01, 0.1, 1, 10]}}}
    with pytest.raises(ConfigError, match="speed mode"):
        load_config(write_yaml(tmp_path / "c.yaml", bad))


def test_models_need_tuning(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="tuning"):
        load_config(write_yaml(tmp_path / "c.yaml", {**VALID, "models": ["lr"]}))


def test_grouped_cv_needs_grouped_split(tmp_path: Path) -> None:
    bad = {**MODELLED, "split": {"method": "random_stratified", "ratios": [0.8, 0.1, 0.1]}}
    with pytest.raises(ConfigError, match="grouped_cv"):
        load_config(write_yaml(tmp_path / "c.yaml", bad))


def test_variants_expand_with_overrides(tmp_path: Path) -> None:
    data = {
        **MODELLED,
        "variants": [
            {"name": "a", "overrides": {"preprocessing": "p_cv"}},
            {"name": "b", "overrides": {"features": {"metadata": True}, "models": ["xgb"]}},
        ],
        "compare_to": "a",
    }
    config = load_config(write_yaml(tmp_path / "c.yaml", data))
    expanded = dict(expand_variants(config))
    assert list(expanded) == ["a", "b"]
    assert expanded["a"].preprocessing == "p_cv" and expanded["a"].models == ["lr"]
    assert expanded["b"].features.metadata is True and expanded["b"].models == ["xgb"]
    assert expanded["b"].preprocessing == "p_cvx" and expanded["b"].variants == []


def test_no_variants_gives_main(tmp_path: Path) -> None:
    config = load_config(write_yaml(tmp_path / "c.yaml", MODELLED))
    assert [name for name, _ in expand_variants(config)] == ["main"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"variants": [{"name": "a"}, {"name": "a"}]}, "unique"),
        ({"variants": [{"name": "A-1"}]}, "only a-z"),
        ({"variants": [{"name": "a", "overrides": {"experiment_id": "x"}}]}, "may not override"),
        ({"variants": [{"name": "a", "overrides": {"sead": 1}}]}, "sead"),
        ({"variants": [{"name": "a"}], "compare_to": "b"}, "compare_to"),
        ({"resample": {"method": "smote", "when": "before_split"}}, "random split"),
        ({"resample": {"method": "copy", "when": "train_only"}}, "resample.method"),
        (
            {"models": ["xgb"], "resample": {"method": "smote", "when": "train_only"}},
            "not supported",
        ),
        ({"stats": {"mcnemar": [["lr"]]}}, "pair"),
    ],
)
def test_invalid_m2_keys_rejected(tmp_path: Path, change: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write_yaml(tmp_path / "c.yaml", {**MODELLED, **change}))


def test_tuning_override_per_model(tmp_path: Path) -> None:
    data = {**MODELLED, "tuning": {**MODELLED["tuning"], "overrides": {"xgb": "validation"}}}
    tuning = load_config(write_yaml(tmp_path / "c.yaml", data)).tuning
    assert tuning.for_model("xgb").method == "validation"
    assert tuning.for_model("lr").method == "grouped_cv"
    bad = {**MODELLED, "tuning": {**MODELLED["tuning"], "overrides": {"bert": "validation"}}}
    with pytest.raises(ConfigError, match="overrides"):
        load_config(write_yaml(tmp_path / "bad.yaml", bad))
