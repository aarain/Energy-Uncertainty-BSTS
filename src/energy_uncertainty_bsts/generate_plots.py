from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_bsts_model(data_series: pd.Series):
    """
    Fits a Bayesian Structural Time Series (BSTS) model and returns the forecast.
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


def main():
    ### Set up paths

    # Go up two levels to reach the project root.
    project_root = Path(__file__).parent.parent.parent

    # TODO: Load data from CSV
    # data_path = project_root / "data" / "processed" / "energy_load.csv"
    assets_dir = project_root / "assets"

    # Ensure the directory exists just in case.
    assets_dir.mkdir(parents=True, exist_ok=True)

    ### Load data

    # TODO: Load data from CSV instead of generating synthetic energy load data.
    # data_frame = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Generate Synthetic Energy Load Data (Norwegian: high variance).
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    load = (
        3000  # base load for NO1 (approx 3000 MW)
        + np.arange(200) * 0.05  # linear trend (mimicking population growth etc.)
        + 300 * np.sin(2 * np.pi * dates.day_of_week / 7 + 1.5)  # 1.5-shifted weekly cycle
        + 1200 * np.cos(2 * np.pi * dates.dayofyear / 366)  # yearly cycle
        + np.random.normal(
            0,
            100,
            len(dates),
        )  # Stochastic noise (mean, standard deviation, days per shock)
        + np.where(
            (dates.month == 1) & (dates.day <= 10), 1000, 0
        )  # simulated shock like the cold snap in January 2024.
    )
    data_frame = pd.DataFrame({"Timestamp": dates, "Load_MW": load})
    data_frame.set_index("Timestamp", inplace=True)

    ### Fit the model
    forecast_data_frame, results = fit_bsts_model(
        data_frame["Load_MW"]  # Use the load column from the DataFrame.
    )

    ### Create plots

    # Create decomposition plot
    fig = results.plot_components(figsize=(12, 10))
    fig.supylabel("Load (MW)")
    plt.suptitle(
        "BSTS Decomposition: NO1 (Oslo) 2024 Energy Load Trends & Seasonality",
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(assets_dir / "bsts_decomposition.png", dpi=300)
    plt.close()
    print("BSTS Decomposition plot saved as assets/bsts_decomposition.png")

    # Create forecast plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        data_frame.index[-60:],
        data_frame["Load_MW"][-60:],
        label="Observed Load",
        color="black",
    )
    plt.plot(
        forecast_data_frame.index,
        forecast_data_frame["mean"],
        label="BSTS Forecast",
        color="blue",
    )
    plt.fill_between(
        forecast_data_frame.index,
        forecast_data_frame["mean_ci_lower"],
        forecast_data_frame["mean_ci_upper"],
        color="blue",
        alpha=0.2,
        label="90% Confidence Interval",
    )
    plt.title("30-Day Energy Demand Probabilistic Forecast")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.savefig(assets_dir / "bsts_forecast.png", dpi=300)
    plt.close()
    print("Forecast plot saved as assets/bsts_forecast.png")


if __name__ == "__main__":
    main()
