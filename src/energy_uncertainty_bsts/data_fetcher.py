import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient


def fetch_nordic_load(year: int = 2026):
    ### Fetch and set the API key

    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key:
        raise ValueError("API Key not found. Ensure ENTSOE_API_KEY is set via .env or environment variables.")

    print(f"Key loaded: {api_key[:5]}...")

    client = EntsoePandasClient(api_key=api_key)

    ### Set metadata, fetch and save load

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year, month=12, day=31, tz="UTC")

    country_code = "NO_1"  # NO1 is the Oslo Price Area

    print(f"Fetching actual load for {country_code} in {year}...")

    load_series = client.query_load(country_code=country_code, start=start, end=end)

    # Most ENTSO-E data is hourly or every 15-min, so resample to 'daily' for the BSTS model.
    daily_load = load_series.resample("D").mean()

    # Save to CSV
    output_path = Path("data/external/nordpool_no1_load.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_load.to_csv(output_path)

    print(f"Success! Data saved to {output_path}")


if __name__ == "__main__":
    fetch_nordic_load()
