import os
import pandas as pd
import pytest

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "hour.csv")

EXPECTED_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit",
    "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt"
]


def test_dataset_file_exists():
    """Verify that dataset/hour.csv exists on disk."""
    assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"


def test_dataset_readable_and_non_empty():
    """Verify that the dataset can be read as a CSV and has records."""
    df = pd.read_csv(DATASET_PATH)
    assert not df.empty, "Dataset dataframe is empty"
    assert len(df) == 17379, f"Expected 17379 rows, found {len(df)}"


def test_dataset_columns_match_schema():
    """Verify that all required columns are present in the dataset."""
    df = pd.read_csv(DATASET_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    """Verify that there are zero missing or null values in the dataset."""
    df = pd.read_csv(DATASET_PATH)
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"


def test_target_column_cnt():
    """Verify that the target column 'cnt' is numeric, positive, and matches casual + registered."""
    df = pd.read_csv(DATASET_PATH)
    assert "cnt" in df.columns
    assert (df["cnt"] > 0).all(), "cnt should be strictly positive"
    # Verification of definition: cnt == casual + registered
    mismatch = (df["cnt"] != (df["casual"] + df["registered"])).sum()
    assert mismatch == 0, f"Found {mismatch} rows where cnt != casual + registered"
