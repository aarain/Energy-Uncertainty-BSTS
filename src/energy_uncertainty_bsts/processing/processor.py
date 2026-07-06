import pandas as pd
from scipy.stats import median_abs_deviation


def preprocess_data(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Take an unprocessed data frame straight from the source and give it a regular structure.

    :param data_frame: A data frame read from some source e.g. CSV.
    :param column: The target column to read load values from.
    :return: The processed data frame.
    """

    df_copy = data_frame.copy()

    if len(df_copy.columns) != 2:
        raise ValueError(f"Expected to preprocess 2 columns, got {len(df_copy.columns)} column(s).")

    # Assume the first column is the timestamp.
    first_col = df_copy.columns[0]
    df_copy = df_copy.set_index(first_col)
    df_copy.index.name = "Timestamp"

    # Remove times and process only dates, before reattaching UTC time.
    df_dates = df_copy.index.astype(str).str.split(" ").str[0]
    df_copy.index = pd.to_datetime(df_dates).normalize().tz_localize("UTC")

    if df_copy.index.freq is None:
        df_copy = df_copy.asfreq("D")

    df_copy = _fill_data_frame(df_copy)  # Fill any missing days.

    # Identify the load column
    if column not in df_copy.columns:
        # If the expected load values column name is not found, take the first available column.
        first_col = df_copy.columns[0]
        print(f"Warning: '{column}' column not found. Using '{first_col}' instead.")
        df_copy = df_copy.rename(columns={first_col: column})

    return df_copy


def remove_outliers(data_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detect and remove outliers to filter out extreme values.

    If a value exceeds the z-score threshold, it is removed and forward-filled.

    The rolling window z-score (Hampel Filter) approach has been chosen to preserve extreme weather events,
    such as a multi-week cold snap, while removing shorter point spikes, e.g. a sensor error.
    Energy market loads typically follow a "random walk with drift", so a rolling window respects this seasonality.
    """

    df_copy = data_frame.copy()

    rolling_window_days = 7  # The number of days the rolling window should have.
    z_score_threshold = 3

    rolling_median = df_copy[column].rolling(window=rolling_window_days, center=True, min_periods=1).median()

    rolling_mad = (
        df_copy[column]
        .rolling(window=rolling_window_days, center=True, min_periods=1)
        .apply(median_abs_deviation, kwargs={"scale": "normal"})
    )

    # TODO: Ensure there are no division by 0 errors: replace 0 with a small epsilon, and replace NaN with the same
    #       epsilon e.g. if the spike occurs at the edge of the dataset, or the window contains only NaNs.
    # rolling_mad = rolling_mad.replace(0, 1e-6).fillna(1e-6)

    modified_z_score = abs((df_copy[column] - rolling_median) / rolling_mad)

    df_copy.loc[modified_z_score > z_score_threshold, column] = None

    df_copy = _fill_data_frame(df_copy)

    return df_copy


def _fill_data_frame(data_frame: pd.DataFrame):
    data_frame = data_frame.ffill()  # Use forward filling to fill the data frame.
    return data_frame
