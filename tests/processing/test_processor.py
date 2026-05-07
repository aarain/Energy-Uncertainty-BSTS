import numpy as np
import pandas as pd

from energy_uncertainty_bsts.processing.processor import remove_outliers
from src.energy_uncertainty_bsts.config import CSV_LOAD_COLUMN_NAME

target_col = CSV_LOAD_COLUMN_NAME


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
