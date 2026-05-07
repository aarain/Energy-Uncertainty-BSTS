import pandas as pd
from scipy.stats import median_abs_deviation


def remove_outliers(data_frame, column) -> pd.DataFrame:
    """
    Detect and remove outliers to filter out extreme values.

    If a value exceeds the z-score threshold, it is removed and forward-filled.

    The rolling window z-score (Hampel Filter) approach has been chosen to preserve extreme weather events,
    such as a multi-week cold snap, while removing shorter point spikes, e.g. a sensor error.
    Energy market loads typically follow a "random walk with drift", so a rolling window respects this seasonality.
    """

    rolling_window_days = 7  # The number of days the rolling window should have.
    z_score_threshold = 3

    df_copy = data_frame.copy()

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
    return df_copy.ffill()
