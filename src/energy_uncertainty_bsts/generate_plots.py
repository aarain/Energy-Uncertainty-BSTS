from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from src.energy_uncertainty_bsts.config import (
    ASSETS_DIR,
    CSV_LOAD_COLUMN_NAME,
    DATA_DIR,
    DECOMPOSITION_FILENAME,
    FORECAST_FILENAME,
    LOAD_DATA_FILENAME,
)
from src.energy_uncertainty_bsts.processor import remove_outliers


def load_and_preprocess_data(data_file: Path, target_col) -> pd.DataFrame:
    """
    Process a CSV file into a data frame.
    """

    if not data_file.exists():
        raise FileNotFoundError(f"Could not find data at {data_file}")

    # Assume the CSV has 2 columns: a date column and a load values column.
    data_frame = pd.read_csv(filepath_or_buffer=data_file, index_col=0)

    # Manually parse dates to adjust for daylight savings by converting the timestamp column to UTC
    data_frame.index = pd.to_datetime(data_frame.index, utc=True)

    if data_frame.index.freq is None:
        data_frame = data_frame.asfreq("D")

    # If there are missing days, forward-fill them so statsmodels doesn't crash.
    data_frame = data_frame.ffill()

    # Identify the load column
    if target_col not in data_frame.columns:
        # If the expected load values column name is not found, take the first available column.
        first_col = data_frame.columns[0]
        print(f"Warning: '{target_col}' column not found. Using '{first_col}' instead.")
        data_frame = data_frame.rename(columns={first_col: target_col})

    data_frame = remove_outliers(data_frame, target_col)

    return data_frame


def fit_bsts_model(data_series: pd.Series):
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


def generate_plots(data_frame, forecast_data_frame, results, assets_dir, target_col):
    """
    Generate two plots, one for the historical decomposition and one for the forecast.
    """

    # Ensure the directory to save the plots to exists
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Create decomposition plot
    fig = results.plot_components(figsize=(12, 10))
    fig.supylabel("Load (MW)")
    plt.suptitle(
        "BSTS Decomposition: NO1 (Oslo) Nordpool Energy Load",
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(assets_dir / DECOMPOSITION_FILENAME, dpi=300)
    plt.close()
    print(f"BSTS Decomposition plot saved as {ASSETS_DIR}{DECOMPOSITION_FILENAME}")

    # Create forecast plot for the last 60 days of real data
    plt.figure(figsize=(10, 6))
    plt.plot(
        data_frame.index[-60:],
        data_frame[target_col][-60:],
        label="Observed Load (Nordpool)",
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
    plt.title("30-Day Energy Demand Forecast (NO1)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.savefig(assets_dir / FORECAST_FILENAME, dpi=300)
    plt.close()
    print(f"Forecast plot saved as {ASSETS_DIR}{FORECAST_FILENAME}")


def main():
    ### Set up paths
    project_root = Path(__file__).parent.parent.parent  # Go up two levels to reach the project root.
    data_path = project_root / f"{DATA_DIR}{LOAD_DATA_FILENAME}"
    assets_dir = project_root / ASSETS_DIR
    target_col = CSV_LOAD_COLUMN_NAME

    data_frame = load_and_preprocess_data(data_path, target_col)

    forecast_data_frame, results = fit_bsts_model(data_frame[target_col])

    generate_plots(data_frame, forecast_data_frame, results, assets_dir, target_col)


if __name__ == "__main__":
    main()
