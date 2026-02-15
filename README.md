# Energy Uncertainty BSTS

This project uses **Bayesian Structural Time Series (BSTS)** to break down energy load into clear trend patterns while accounting for external market shocks. Unlike traditional point-estimate models, this approach uses a **probabilistic forecast** to quantify "worst-case" scenarios. It is specifically designed to handle volatile global energy markets, providing a more reliable edge for short-term trading decisions.

## 📈 Model Output

Historic trend decomposition - _Isolates the underlying growth trend from weekly seasonality and irregular market volatility:_
![BSTS decomposition](assets/bsts_decomposition.png)

Point estimate and confidence intervals - _Quantifies 90% confidence intervals_ (P10/P90) to inform Value-at-Risk (VaR):
![BSTS forecast](assets/bsts_forecast.png)

## 📊 Statistical Profile

The model is built on a **Linear Gaussian State Space** framework. Energy markets often exhibit non-linear behavior, but using BSTS provides a reliable baseline by decomposing the signal as follows:

* **Distribution**: A Gaussian (Normal) error structure is assumed for the stochastic components. This allows generation of the symmetric 90% predictive intervals seen in the forecast.

* **Components**:
  * **Local Level**: A random walk process that captures the shifting baseline of energy demand.
  * **Weekly Seasonality**: This (periodically) accounts for the differences in consumption patterns between weekdays and weekends.

* **Bayesian Inference**: Unlike models that provide a single "best" fit, this model uses weighted integration to account for parameter uncertainty, resulting in more realistic estimation of maximum potential loss (VaR).

## 🛠 Installation

How to get the development env running:

1. **Clone the repo**, e.g. by using the GitHub CLI: `gh repo clone aarain/Energy-Uncertainty-BSTS`
2. **Set up the virtual environment**:
    ```bash
   python3 -m venv .venv --system-site-packages
   source .venv/bin/activate
   pip install -r requirements.txt
    ```

### Development Workflow
This project uses `pip-tools` to manage dependencies. To update the lockfile:
1. Ensure your environment has `pip-tools` installed.
2. Run `pip-compile requirements.in` to generate a fresh `requirements.txt`.
3. Run `pip-sync` to align your virtual environment.

**Note**: This project requires `scipy < 1.13.0` to maintain compatibility with the `statsmodels` state-space backend in Python 3.12.

## ▶️ Usage

**Generate Plots**: `python src/energy_uncertainty_bsts/generate_plots.py`

**Run Linter**: `ruff check .`

**Run Tests**: `pytest`

## 🚀 Roadmap

### Current project status

The current version provides a functional probabilistic baseline for energy load. This implementation:

* Utilises Bayesian Structural Time Series (BSTS) to isolate the local Level (long-term trend) from weekly seasonality (7-day cycles).
* Moves beyond single-point estimates by generating 90% confidence intervals, providing a mathematical basis for Value-at-Risk (VaR) analysis.
* Uses a high-variance synthetic data engine to simulate "Dubai-style" energy shocks.

### Future development

To move towards a production-ready trading tool, the following phases are planned:

#### Phase 2: Real-world data & sanitisation

* Replace synthetic data generation with real historical CSV datasets:
   * This should add realistic external regressors such as weather (humidity and temperature), tariff tiers (price increasing with consumption), and Ramadan (holidays).
   * Use either DEWA or SEWA (Dubai/Sharjah Electricity and Water Authority) data.
* Sanitise the (real-world) input data:
   * Handle sensor gaps using forward-filling/linear interpolation.
   * Detect and remove outliers caused by predictable or extreme events e.g. grid maintenance or extreme heatwave.

#### Phase 3: Model validation

* Validate the sanitised data:
   * Recursively back-test historical data by implementing a Time Series split using a rolling window to measure the data's MASE (Mean Absolute Scaled Error).
   * Verify the model's calibration a PIT (Probability Integral Transform) histogram. A U-shaped histogram implies underconfidence (narrow intervals), while an inverted U shape implies overconfidence (wide intervals).

#### Phase 4: Machine Learning Operations & interactive visualisation

* Use GitHub Actions to automate the process of retraining the model on new seasonal (weekly) data.
* Build an interactive dashboard (e.g. with Streamlit). This should allow the user to:
   * Recalculate the BSTS forecast in real time based on certain parameters e.g. humidity.
   * Toggle between different VaR levels i.e. 90%, 95%, 99%.
   * Compare the sanitised input data with their model's calibrated forecast.
