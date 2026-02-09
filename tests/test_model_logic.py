import numpy as np
import pandas as pd

from energy_uncertainty_bsts.generate_plots import fit_bsts_model


def test_forecast_horizon():
    """
    Ensure the model produces the requested 30-day forecast window.
    """

    data = pd.Series(np.random.normal(50, 2, 50))
    forecast, _ = fit_bsts_model(data)

    assert len(forecast) == 30, f"Expected 30 days of forecast, got {len(forecast)}."


def test_bsts_forecast_integrity():
    """
    Validate that the P90/P10 forecast intervals are mathematically consistent and capturing uncertainty.
    """

    # 1. Create synthetic energy load data
    np.random.seed(42)
    data = np.random.normal(100, 10, 100)
    df_series = pd.Series(data)

    # 2. Fit model and get forecast summary frame
    # (Assuming fit_bsts_model returns the summary_frame with 'mean', 'mean_ci_lower', 'mean_ci_upper')
    forecast, _ = fit_bsts_model(df_series)

    # 3. Assertions for Statistical Sanity
    assert not forecast.isnull().values.any(), "Forecast contains NaN values."

    # Check that lower < mean < upper
    assert (forecast["mean_ci_upper"] > forecast["mean_ci_lower"]).all(), (
        "Confidence intervals are inverted."
    )
    assert (forecast["mean_ci_upper"] >= forecast["mean"]).all(), (
        "Mean exceeds upper confidence bound"
    )
    assert (forecast["mean"] >= forecast["mean_ci_lower"]).all(), (
        "Mean is below lower confidence bound"
    )

    # Check that uncertainty is non-zero (The model is actually 'uncertain')
    spread = forecast["mean_ci_upper"] - forecast["mean_ci_lower"]
    assert np.all(spread > 0), (
        "Model is producing zero-width confidence intervals (not capturing uncertainty)."
    )
