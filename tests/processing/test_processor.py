import re

import numpy as np
import pandas as pd
import pytest

from energy_uncertainty_bsts.config import CSV_LOAD_COLUMN_NAME
from energy_uncertainty_bsts.processing.processor import preprocess_data, remove_outliers

target_col = CSV_LOAD_COLUMN_NAME


@pytest.fixture
def test_data_frame() -> pd.DataFrame:
    """
    Generate a test data frame.
    """

    data = {
        "": [
            "2026-01-01 00:00:00+01:00",
            "2026-01-02 00:00:00+01:00",
            # Note the missing January 3rd
            "2026-01-04 00:00:00+02:00",
        ],
        target_col: [5000.0, 5100.0, 5300.0],
    }

    return pd.DataFrame(data)


def test_preprocess_data_verifies_two_columns():
    df_one_clm = pd.DataFrame(
        {
            target_col: [100.0, 100.0],
        }
    )

    expected_err_msg = f"Expected to preprocess 2 columns, got {len(df_one_clm.columns)} column(s)."
    with pytest.raises(ValueError, match=re.escape(expected_err_msg)):
        preprocess_data(df_one_clm, target_col)


def test_preprocess_data_sets_timestamp_index(test_data_frame):
    data_frame = preprocess_data(test_data_frame, target_col)

    assert data_frame.index.name == "Timestamp"
    assert "Timestamp" not in data_frame.columns
    assert len(data_frame.columns) == 1


def test_preprocess_data_utc_conversion(test_data_frame):
    data_frame = preprocess_data(test_data_frame, target_col)

    assert str(data_frame.index.tz) == "UTC"


def test_preprocess_data_sets_frequency(test_data_frame):
    data_frame = preprocess_data(test_data_frame, target_col)

    assert data_frame.index.freqstr == "D"


def test_preprocess_data_fills_missing_days(test_data_frame):
    data_frame = preprocess_data(test_data_frame, target_col)

    # The missing January 3rd date should be filled with Jan 2nd's value
    assert len(data_frame) == 4

    target_timestamp = pd.Timestamp(ts_input="2026-01-03", tz="UTC")
    assert data_frame.loc[target_timestamp, target_col] == 5100.0


def test_preprocess_data_column_fallback(test_data_frame):
    bad_col_data_frame = test_data_frame.rename(columns={"Timestamp": "BadColumnName"})

    data_frame = preprocess_data(bad_col_data_frame, target_col)

    # Assert that the data frame renames its bad column name to the expected column name.
    assert target_col in data_frame.columns
    assert data_frame[target_col].iloc[0] == test_data_frame[target_col].iloc[0]


def test_remove_outliers_detects_spike():
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    load = [100.0] * 20
    load[10] = 5000.0  # Insert one large value to simulate the spike

    original_data_frame = pd.DataFrame({target_col: load}, index=dates)

    cleaned_data_frame = remove_outliers(original_data_frame, target_col)

    assert cleaned_data_frame[target_col].max() <= 120
    assert cleaned_data_frame[target_col].iloc[10] == 100.0
    assert not cleaned_data_frame[target_col].isnull().any()


def test_remove_outliers_ignores_trend():
    load = np.linspace(100, 120, 20)  # A linear increase for 20 days from 100 to 120.
    dates = pd.date_range("2026-01-01", periods=20, freq="D")

    original_data_frame = pd.DataFrame(data={target_col: load}, index=dates)

    cleaned_data_frame = remove_outliers(original_data_frame, target_col)

    pd.testing.assert_frame_equal(cleaned_data_frame, original_data_frame)
