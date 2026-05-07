from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pandas import DataFrame

from energy_uncertainty_bsts.visualisation.generate_plots import get_project_root, load_and_preprocess_data
from src.energy_uncertainty_bsts.config import CSV_LOAD_COLUMN_NAME

target_col = CSV_LOAD_COLUMN_NAME


@pytest.fixture
def fake_data_frame() -> DataFrame:
    """
    Generate a test data frame.
    """

    data = {
        "Timestamp": ["2026-01-01", "2026-01-02", "2026-01-04"],  # Note the missing January 3rd
        target_col: [100.0, 110.0, 130.0],
    }

    return pd.DataFrame(data)


@pytest.fixture
def fake_csv(tmp_path, fake_data_frame) -> Path:
    """
    Creates a temporary CSV file for testing.
    """

    csv_path = tmp_path / "test_load.csv"
    data_frame = fake_data_frame
    data_frame.to_csv(csv_path, index=False)

    return csv_path


def test_load_and_preprocess_file_not_found():
    """
    Verify explicit error handling for missing files.
    """

    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data(Path("non_existent.csv"), target_col)


def test_load_and_preprocess_utc_conversion(fake_csv):
    """
    Verify timezones are converted to UTC.
    """

    data_frame = load_and_preprocess_data(fake_csv, target_col)
    assert str(data_frame.index.tz) == "UTC"


def test_load_and_preprocess_fills_missing_days(fake_csv):
    """
    Verify that the frequency is set and that gaps in the timeline are filled.
    """

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

    data_frame = load_and_preprocess_data(csv_path, target_col)

    # Assert that the data frame renames its bad column name to the expected column name.
    assert target_col in data_frame.columns
    assert data_frame[target_col].iloc[0] == 1


def test_load_and_preprocess_calls_outlier_removal(fake_data_frame, fake_csv):
    """
    Verify that the function calls the remove_outliers function.
    """

    with patch("energy_uncertainty_bsts.visualisation.generate_plots.remove_outliers") as mock_remove:
        # The main function will crash if the mock does not return a data frame.
        mock_remove.return_value = fake_data_frame

        load_and_preprocess_data(fake_csv, target_col)

        mock_remove.assert_called_once()

        # Check that it was called with the right column name.
        # The first argument (0) is the dataframe, the second (1) is the column name.
        args, kwargs = mock_remove.call_args
        assert args[1] == target_col


def test_get_project_root_success():
    project_root = "Energy-Uncertainty-BSTS"
    _, separator, after = str(get_project_root()).partition(project_root)

    assert f"{separator}{after}" == project_root


def test_get_project_root_error():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            get_project_root()
