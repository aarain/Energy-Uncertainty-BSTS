import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. Generate Synthetic Energy Load Data (Dubai-style: High Variance)
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
# Trend + Weekly Cycle + Humidity Shock + Noise
load = (
    100
    + np.arange(200) * 0.2
    + 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    + np.random.normal(0, 5, 200)
)
df = pd.DataFrame({"ds": dates, "y": load}).set_index("ds")

# 2. Fit the BSTS (Unobserved Components) Model
# 'local level' = Trend, 'seasonal=7' = Weekly patterns
model = sm.tsa.UnobservedComponents(df["y"], level="local level", seasonal=7)
res = model.fit(disp=False)

# 3. Create the Professional Plot
fig = res.plot_components(figsize=(12, 10))
plt.suptitle("BSTS Decomposition: Energy Load Trends & Seasonality", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 4. Save for GitHub
plt.savefig("assets/bsts_decomposition.png", dpi=300)
print("Success! BSTS Decomposition saved as bsts_decomposition.png")

# 5. Generate a Forecast Plot with Confidence Intervals
forecast = res.get_forecast(steps=30)  # 30-day forecast
mean_forecast = forecast.summary_frame()["mean"]
conf_int = forecast.summary_frame(alpha=0.10)  # 90% Confidence Interval

plt.figure(figsize=(10, 5))
plt.plot(df.index[-50:], df["y"][-50:], label="Observed Load", color="black")
plt.plot(mean_forecast.index, mean_forecast, label="BSTS Forecast", color="blue")
plt.fill_between(
    conf_int.index,
    conf_int["mean_ci_lower"],
    conf_int["mean_ci_upper"],
    color="blue",
    alpha=0.2,
    label="90% Confidence Interval",
)
plt.title("30-Day Energy Demand Probabilistic Forecast")
plt.legend()
plt.savefig("assets/bsts_forecast.png", dpi=300)
print("Forecast plot saved as bsts_forecast.png")
