"""PRD M6 checks: export writes every Section 10.6 file; loading refuses on a hash,
version or smoke-score mismatch."""

import shutil
from pathlib import Path

import pytest

from cvx.artifacts.load import MODEL_CARD, SMOKE_FILE, ModelStoreError, file_hash, load_store
from cvx.artifacts.save import parse_model_id
from cvx.runio import read_json, write_json


def copy_store(source: Path, tmp_path: Path) -> Path:
    """A private copy of the session store that a test may damage."""
    target = tmp_path / "store"
    shutil.copytree(source, target)
    return target


def test_parse_model_id() -> None:
    assert parse_model_id("lr_full") == ("lr", "full")
    assert parse_model_id("linear_svm_text") == ("linear_svm", "text")
    with pytest.raises(ValueError):
        parse_model_id("bert_full")


def test_export_writes_every_file(tiny_store: Path) -> None:
    card = read_json(tiny_store / MODEL_CARD)
    for model_id in ("lr_text", "lr_full", "xgb_text", "xgb_full"):
        files = card["models"][model_id]["files"]
        assert {"model", "features", "feature_means"} <= set(files)
        assert ("booster" in files) == model_id.startswith("xgb")
        for name in files.values():
            assert (tiny_store / name).is_file() and card["sha256"][name] == file_hash(
                tiny_store / name
            )
        entry = card["models"][model_id]
        assert (
            set(entry["thresholds"]) == {"t_f1", "t_r90"}
            and "threshold_free" in entry["test_metrics"]
        )
        assert entry["input_columns"][0] == "text"
    assert {"seed", "dataset", "git", "libraries", "source_run"} <= set(card)
    smoke = read_json(tiny_store / SMOKE_FILE)
    assert len(smoke["rows"]) == 5 and set(smoke["scores"]) == set(card["models"])


def test_load_store_ok(tiny_store: Path) -> None:
    store = load_store(tiny_store)
    assert set(store.models) == {"lr_text", "lr_full", "xgb_text", "xgb_full"}
    assert store.report["status"] == "ok"
    assert store.models["lr_full"].means is not None


def test_refuses_changed_file(tiny_store: Path, tmp_path: Path) -> None:
    store = copy_store(tiny_store, tmp_path)
    with (store / "lr_text.pkl").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ModelStoreError, match="SHA-256"):
        load_store(store)


def test_refuses_version_mismatch(tiny_store: Path, tmp_path: Path) -> None:
    store = copy_store(tiny_store, tmp_path)
    card = read_json(store / MODEL_CARD)
    card["libraries"]["scikit-learn"] = "0.1.0"
    write_json(store / MODEL_CARD, card)
    with pytest.raises(ModelStoreError, match="versions differ"):
        load_store(store)


def test_refuses_smoke_mismatch(tiny_store: Path, tmp_path: Path) -> None:
    store = copy_store(tiny_store, tmp_path)
    smoke = read_json(store / SMOKE_FILE)
    smoke["scores"]["lr_text"][0] += 0.01
    write_json(store / SMOKE_FILE, smoke)
    card = read_json(store / MODEL_CARD)
    card["sha256"][SMOKE_FILE] = file_hash(store / SMOKE_FILE)  # hash ok, score wrong
    write_json(store / MODEL_CARD, card)
    with pytest.raises(ModelStoreError, match="Smoke"):
        load_store(store)


def test_refuses_unlisted_model_and_missing_store(tiny_store: Path, tmp_path: Path) -> None:
    with pytest.raises(ModelStoreError, match="not in"):
        load_store(tiny_store, ["nb_text"])
    with pytest.raises(ModelStoreError, match="Export models first"):
        load_store(tmp_path / "empty")
