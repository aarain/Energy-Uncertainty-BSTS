import pandas as pd
import statsmodels.api as sm


def fit_sts_model(data_series: pd.Series):
    """
    Fits a State Space Structural Time Series (STS) model using Maximum Likelihood Estimation (MLE)
    and returns the probabilistic forecast.
    """

    # Define the unobserved components model (Trend and Seasonality).
    model = sm.tsa.UnobservedComponents(
        data_series,
        level="local level",  # Trend
        seasonal=7,  # Assume weekly seasonality (patterns) for energy load. Patterns over the week sum to ~0.
    )
    results = model.fit(disp=False)

    # Set 30-day forecast and a 90% confidence interval
    forecast_data_frame = results.get_forecast(steps=30).summary_frame(alpha=0.10)

    return forecast_data_frame, results
