# Quantitative Finance: Portfolio Risk, Volatility, VaR, Option Pricing & Hedging

## Overview

This project looks at a few important topics in quantitative finance, starting from portfolio analysis and moving towards risk management and option pricing.

The portfolio consists of five different assets:

* **HDFC Asset Management Company** — Large-cap equity
* **NIFTY 50** — Market index
* **EUR/INR** — Foreign exchange
* **Macpower CNC Machines** — Small-cap equity
* **Gold** — Commodity

Historical price data is used to calculate returns, study volatility, estimate Value at Risk, and test the performance of different risk models. The project also includes a separate section on European option pricing using simulated GBM paths, calculation of option Greeks, and a simple delta-neutral hedging strategy.

The data for the portfolio analysis covers **June 2019 to June 2024**.
This project closely aligns with the assignment of FRAM course of BITS Pilani Hyderabad Campus and a few contribuitions were made from my side (correcting some statistical tests and delta hedging).

---

## What the Project Covers

### 1. Portfolio Data & Preprocessing

Historical adjusted closing prices are downloaded using `yfinance`.

The data is cleaned using forward filling, after which daily log returns are calculated:

```math
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
```

The notebook also looks at the price and return series through different plots and calculates basic statistics such as:

* Mean
* Standard deviation
* Skewness
* Kurtosis

The portfolio starts with equal weights of **20% for each asset**.

---

### 2. Volatility Modeling

The portfolio returns are first modeled using an ARIMA model. Different specifications are compared to select a suitable model.

The residuals are then checked using:

* ACF and PACF
* Breusch–Pagan test
* Goldfeld–Quandt test
* Ljung–Box test on squared residuals

Several GARCH-type models are fitted and compared using AIC and BIC:

* GARCH
* EGARCH
* GJR-GARCH
* APARCH
* FIGARCH

The selected model is then used to forecast volatility for the next **30 days**.

---

### 3. Value at Risk (VaR)

VaR is estimated using three different approaches:

**Historical Simulation**
Uses bootstrap resampling of the historical returns.

**Variance-Covariance Method**
Uses the estimated mean and standard deviation of returns along with the normal distribution.

**Monte Carlo Simulation**
Fits different distributions to the return data and uses the selected distribution to simulate future returns.

The VaR estimates are calculated at both **95% and 99% confidence levels**, for one-day as well as ten-day horizons.

---

### 4. VaR Backtesting

The VaR estimates are also tested using:

* **Kupiec POF test**
* **Christoffersen test**

These tests are used to check whether the observed VaR violations are consistent with what the model predicts.

---

### 5. European Option Pricing

The project also includes a small option-pricing component using **SBI** stock data.

Stock-price paths are simulated using Geometric Brownian Motion:

```math
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t.
```

The simulated paths are then used to estimate European call and put prices.

Finite differences are used to estimate:

* Delta
* Gamma
* Vega
* Theta

The calculations are repeated for different strike prices and maturities.

---

### 6. Delta-Neutral Hedging

The final section looks at a simple delta-neutral portfolio strategy.

The assumed delta exposures are:

| Asset        | Delta |
| ------------ | ----: |
| HDFC AMC     |     1 |
| NIFTY 50     |     1 |
| EUR/INR      |    -1 |
| Macpower CNC |     1 |
| Gold         |   0.5 |

The portfolio weights are rebalanced so that:

```math
\sum_i w_i\Delta_i = 0
```

while keeping the portfolio fully invested:

```math
\sum_i w_i = 1.
```

Short selling is not allowed, so each weight is restricted to:

```math
0 \leq w_i \leq 1.
```

The resulting returns are then compared with the original portfolio.

---

## Libraries Used

The analysis is implemented in Python using:

* `yfinance`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scipy`
* `statsmodels`
* `pmdarima`
* `arch`

---

## Notebook

The complete analysis and code are available in:

```text
quantitative_finance_risk_analysis.ipynb
```

---

## Project Structure

```text
Quantitative_Finance_Risk_Analysis/
│
├── README.md
└── quantitative_finance_risk_analysis.ipynb
```

---

## Disclaimer

This project is for academic and educational purposes. The results are based on the assumptions and models used in the notebook and should not be taken as investment advice.
