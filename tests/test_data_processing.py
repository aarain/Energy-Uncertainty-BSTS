from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from energy_uncertainty_bsts.config import CSV_LOAD_COLUMN_NAME
from energy_uncertainty_bsts.visualisation.generate_plots import get_project_root, load_and_preprocess_data

target_col = CSV_LOAD_COLUMN_NAME


@pytest.fixture
def fake_data_frame() -> pd.DataFrame:
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
    root_path = get_project_root()

    assert root_path.is_dir()
    # Assert the root path is correct by verifying an expected root-level file or directory exists.
    assert (root_path / "pyproject.toml").exists() or (root_path / "src").exists()
    assert root_path.name == "Energy-Uncertainty-BSTS"


def test_get_project_root_error():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            get_project_root()
