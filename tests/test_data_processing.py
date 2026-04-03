from pathlib import Path

import pandas as pd
import pytest

from src.energy_uncertainty_bsts.config import CSV_LOAD_COLUMN_NAME
from src.energy_uncertainty_bsts.generate_plots import load_and_preprocess_data


@pytest.fixture
def fake_csv(tmp_path):
    """
    Creates a temporary CSV file for testing.
    """

    csv_path = tmp_path / "test_load.csv"
    data = {
        "Timestamp": ["2026-01-01", "2026-01-02", "2026-01-04"],  # Note the missing January 3rd
        CSV_LOAD_COLUMN_NAME: [100.0, 110.0, 130.0],
    }
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(csv_path, index=False)

    return csv_path


def test_load_and_preprocess_file_not_found():
    """
    Verify explicit error handling for missing files.
    """

    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data(Path("non_existent.csv"), CSV_LOAD_COLUMN_NAME)


def test_load_and_preprocess_utc_conversion(fake_csv):
    """
    Verify timezones are converted to UTC.
    """

    data_frame = load_and_preprocess_data(fake_csv, CSV_LOAD_COLUMN_NAME)
    assert str(data_frame.index.tz) == "UTC"


def test_load_and_preprocess_fills_missing_days(fake_csv):
    """
    Verify that the frequency is set and that gaps in the timeline are filled.
    """

    target_col = CSV_LOAD_COLUMN_NAME
    data_frame = load_and_preprocess_data(fake_csv, target_col)

    assert data_frame.index.freqstr == "D"

    # The missing January 3rd date should be filled with Jan 2nd's value (110.0)
    assert len(data_frame) == 4
    assert data_frame.loc["2026-01-03", target_col] == 110.0


def test_load_and_preprocess_column_fallback(tmp_path):
    """
    Ensure that if the expected load values column name is not found, the first available column is used.
    """

    csv_path = tmp_path / "bad_col.csv"
    pd.DataFrame({"BadColumnName": [1, 2, 3]}, index=["2026-01-01", "2026-01-02", "2026-01-03"]).to_csv(csv_path)

    target_col = CSV_LOAD_COLUMN_NAME
    data_frame = load_and_preprocess_data(csv_path, target_col)

    # Assert that the data frame renames its bad column name to the expected column name.
    assert target_col in data_frame.columns
    assert data_frame[target_col].iloc[0] == 1
