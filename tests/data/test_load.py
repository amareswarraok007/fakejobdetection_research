"""Tests for cvx.data.load: clear errors, column checks, hashing, missing detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvx.data.load import DataFileMissingError, file_sha256, is_missing, load_postings


def test_missing_csv_gives_clear_message(tmp_path: Path) -> None:
    missing = tmp_path / "fake_job_postings.csv"
    with pytest.raises(DataFileMissingError) as info:
        load_postings(missing)
    message = str(info.value)
    assert str(missing) in message
    assert "kaggle.com" in message


def test_loads_valid_csv(tmp_path: Path, tiny_postings: pd.DataFrame) -> None:
    path = tmp_path / "data.csv"
    tiny_postings.to_csv(path, index=False)
    df = load_postings(path)
    assert len(df) == 6
    assert df["fraudulent"].tolist() == [0, 0, 1, 1, 0, 1]
    assert pd.isna(df.loc[0, "benefits"])  # missing values stay NaN at load time


def test_missing_column_rejected(tmp_path: Path, tiny_postings: pd.DataFrame) -> None:
    path = tmp_path / "data.csv"
    tiny_postings.drop(columns=["benefits"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="benefits"):
        load_postings(path)


def test_bad_label_rejected(tmp_path: Path, tiny_postings: pd.DataFrame) -> None:
    path = tmp_path / "data.csv"
    tiny_postings.assign(fraudulent=[0, 0, 1, 2, 0, 1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not 0/1"):
        load_postings(path)


def test_file_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    # Standard SHA-256 test vector for the bytes "abc".
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(path) == expected


def test_is_missing() -> None:
    series = pd.Series([np.nan, "", "   ", "x", 0, None], dtype=object)
    assert is_missing(series).tolist() == [True, True, True, False, False, True]
