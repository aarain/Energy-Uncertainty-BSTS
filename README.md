# Energy Uncertainty BSTS

This project uses **Bayesian Structural Time Series (BSTS)** to break down energy load into clear trend patterns while accounting for external market shocks. Unlike traditional point-estimate models, this approach uses a **probabilistic forecast** to quantify "worst-case" scenarios. It is specifically designed to handle volatile global energy markets, providing a more reliable edge for short-term trading decisions.

## 📊 Model Output

Historic trend decomposition - _Isolates the underlying growth trend from weekly seasonality and irregular market volatility:_
![BSTS decomposition](assets/bsts_decomposition.png)

Point estimate and confidence intervals - _Quantifies 90% confidence intervals_ (P10/P90) to inform Value-at-Risk (VaR):
![BSTS forecast](assets/bsts_forecast.png)

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

## 🚀 Usage

**Generate Plots**: `python src/energy_uncertainty_bsts/generate_plots.py`

**Run Linter**: `ruff check .`

**Run Tests**: `pytest`
