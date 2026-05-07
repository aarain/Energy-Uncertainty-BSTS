import numpy as np
import pandas as pd
import pytest

from energy_uncertainty_bsts.modelling.sts_modeller import fit_sts_model


@pytest.fixture
def generate_energy_load(seed=42, size=200, mean=100, std=10):
    """
    Generates reproducible normal distribution data.
    """

    rng = np.random.default_rng(seed)
    data = rng.normal(loc=mean, scale=std, size=size)
    df_series = pd.Series(data)
    return df_series


def test_forecast_horizon(generate_energy_load):
    """
    Ensure the model produces the requested 30-day forecast window.
    """

    forecast, _ = fit_sts_model(generate_energy_load)

    assert len(forecast) == 30, f"Expected 30 days of forecast, got {len(forecast)}."


def test_sts_forecast_integrity(generate_energy_load):
    """
    Validate that the P90/P10 forecast intervals are mathematically consistent and capturing uncertainty.
    """

    forecast, _ = fit_sts_model(generate_energy_load)

    assert not forecast.isnull().values.any(), "Forecast contains NaN values."

    # Check that lower < mean < upper
    assert (forecast["mean_ci_upper"] > forecast["mean_ci_lower"]).all(), "Confidence intervals are inverted."
    assert (forecast["mean_ci_upper"] >= forecast["mean"]).all(), "Mean exceeds upper confidence bound"
    assert (forecast["mean"] >= forecast["mean_ci_lower"]).all(), "Mean is below lower confidence bound"

    # Check that uncertainty is non-zero (The model is actually 'uncertain')
    spread = forecast["mean_ci_upper"] - forecast["mean_ci_lower"]
    assert np.all(spread > 0), "Model is producing zero-width confidence intervals (not capturing uncertainty)."


def test_forecast_variance_expansion(generate_energy_load):
    """
    Check that uncertainty increases over time.
    """

    forecast, _ = fit_sts_model(generate_energy_load)

    initial_spread = forecast["mean_ci_upper"].iloc[0] - forecast["mean_ci_lower"].iloc[0]
    final_spread = forecast["mean_ci_upper"].iloc[-1] - forecast["mean_ci_lower"].iloc[-1]

    assert final_spread >= initial_spread, "Uncertainty should not decrease when forecast into the future."


def test_sts_residual_normality(generate_energy_load):
    """
    Check that standardised residuals (of a Gaussian State Space model) should have a mean near 0
     (be approximately normally distributed).
    """

    _, results = fit_sts_model(generate_energy_load)

    # Use standardised innovation residuals since the raw residual mean includes a local level and seasonality
    # resulting in the possibility of large errors in the first few steps (seven days).
    standardised_residuals = results.standardized_forecasts_error[0][7:]

    residual_mean = np.mean(standardised_residuals)
    assert pytest.approx(residual_mean, abs=1.0) == 0, f"Residual mean {residual_mean} is too far from zero."


def test_sts_parameter_stability(generate_energy_load):
    """
    Ensure the model correctly identifies the 'local level' (true demand for energy) variance as a positive number.
    """

    _, results = fit_sts_model(generate_energy_load)

    # Since sigma2.level is the variance of the trend component, this being negative means that to calculate
    # VaR = mean + (z-score * standard_deviation) where standard_deviation = sqrt(sigma2.level) collapses for -ve
    # sigma2.level.
    level_variance = results.params["sigma2.level"]
    assert level_variance >= 0, "Model produced negative variance: check state-space constraints."
