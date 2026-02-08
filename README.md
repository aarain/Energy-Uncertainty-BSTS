# Energy Uncertainty BSTS

This project uses **Bayesian Structural Time Series (BSTS)** to break down energy load into clear trend patterns while accounting for external market shocks. Unlike traditional point-estimate models, this approach uses a **probabilistic forecast** to quantify "worst-case" scenarios. It is specifically designed to handle volatile global energy markets, providing a more reliable edge for short-term trading decisions.

## 📊 Model Output

Historic trend decomposition - _Isolates the underlying growth trend from weekly seasonality and irregular market volatility:_
![BSTS decomposition](assets/bsts_decomposition.png)

Point estimate and confidence intervals - _Quantifies 90% confidence intervals_:
![BSTS forecast](assets/bsts_forecast.png)

## 🚀 Installation

How to get the development env running:

1. **Clone the repo**, e.g. by using the GitHub CLI: `gh repo clone aarain/Energy-Uncertainty-BSTS`
2. **Set up the environment**:
    ```bash
   python3 -m venv .venv --system-site-packages
   source .venv/bin/activate
   pip install -r requirements.txt
    ```

## 🛠 Usage

**Generate Plots**: `python src/energy_uncertainty_bsts/generate_plots.py`

**Run Linter**: `ruff check .`

**Run Tests**: `pytest`

**Note**: This project requires `scipy < 1.13.0` to maintain compatibility with the `statsmodels` state-space backend in Python 3.12.
