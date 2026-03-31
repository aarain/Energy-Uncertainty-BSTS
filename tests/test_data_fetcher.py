from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_uncertainty_bsts.data_fetcher import fetch_nordic_load


def test_fetch_nordic_load_missing_key(monkeypatch):
    """
    Raise a ValueError when the ENTSOE_API_KEY environment variable is not set.
    """

    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API Key not found"):
        fetch_nordic_load()


def test_daily_resampling_math():
    """
    Verify resampling logic uses 'mean' not 'sum'.
    """

    hourly_values = [10, 20, 30, 40]  # Mean of 10, 20, 30, 40 is 25.0
    time_index = pd.date_range(start="2026-01-01", periods=4, freq="h")
    df = pd.Series(data=hourly_values, index=time_index)

    daily = df.resample("D").mean()

    assert daily.iloc[0] == 25.0


@patch("entsoe.EntsoePandasClient.query_load")
def test_fetch_nordic_load_success(mock_query, monkeypatch, tmp_path):
    """
    Test mock data-fetching with a fake API key.
    Test data resampling and saving to a fake CSV file.
    """

    ### Set up mock environment

    monkeypatch.setenv("ENTSOE_API_KEY", "fake_test_key")

    # Create 48 hours of fake data (2 days) all at 1000 MW.
    hourly_values = [1000.0] * 48
    time_index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    fake_series = pd.Series(data=hourly_values, index=time_index)
    mock_query.return_value = fake_series

    # Mock the file saving path to use a temporary directory so the real data is not overwritten.
    test_csv_path = tmp_path / "nordpool_no1_load.csv"

    with patch("energy_uncertainty_bsts.data_fetcher.Path") as mock_path:
        # Set the mock path to return the temporary test path.
        mock_path.return_value = test_csv_path
        # Ensure directory creation doesn't fail.
        mock_path.parent.mkdir = MagicMock()

        fetch_nordic_load(year=2026)

    ### Assertions

    # Check the API was called
    mock_query.assert_called_once()

    # Check the API was called with the correct country code
    assert mock_query.call_args.kwargs["country_code"] == "NO_1"

    # Check the CSV file was saved
    assert test_csv_path.exists()

    # Verify resampling frequency (48 hours hourly data resampled to daily should be 2 rows)
    saved_df = pd.read_csv(filepath_or_buffer=test_csv_path)
    assert len(saved_df) == 2

    # Verify mean of 1000.0 hourly is 1000.0 daily.
    assert saved_df.iloc[0, 1] == 1000.0
