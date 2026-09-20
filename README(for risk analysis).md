**Quantitative Finance: Portfolio Risk, Volatility, VaR, Option Pricing & Hedging**

This notebook walks through a practical quantitative-finance workflow. It starts with a small multi-asset portfolio, studies its risk characteristics, builds volatility models, estimates Value-at-Risk, and finishes with a simple European-option pricing exercise plus a delta-neutral hedge.

The portfolio contains five assets chosen for diversity:

- **HDFC AMC** – large-cap equity  
- **Macpower CNC** – small-cap equity  
- **NIFTY 50** – market index  
- **EUR/INR** – FX rate  
- **Gold** – commodity (converted to INR per 10 g)

Price data run from mid-June 2019 to mid-June 2024. The work loosely follows the FRAM assignment at BITS Pilani Hyderabad; a few statistical tests and the delta-hedging section were cleaned up or extended along the way.

---

### 1. Data & Pre-processing

Adjusted closes are pulled with `yfinance`. Missing values are forward-filled, then daily log returns are formed:

\[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
\]

Basic descriptive statistics (mean, volatility, skewness, kurtosis) and a handful of time-series / histogram plots give a first look at the data. The starting point is an equal-weighted portfolio (20 % each).

---

### 2. Volatility Modelling

An ARIMA model is first fitted to the portfolio returns. Residuals are inspected with ACF/PACF plots and the usual battery of tests (Breusch–Pagan, Goldfeld–Quandt, Ljung–Box on squared residuals).

Several GARCH-family specifications are then estimated and ranked by AIC/BIC:

- GARCH  
- EGARCH  
- GJR-GARCH  
- APARCH  
- FIGARCH  

The preferred model is used to produce a 30-day volatility forecast.

---

### 3. Value-at-Risk

Three VaR estimators are compared at the 95 % and 99 % levels, for both 1-day and 10-day horizons:

- **Historical simulation** – bootstrap of the observed returns  
- **Variance–covariance** – parametric normal approximation  
- **Monte-Carlo** – draws from a distribution fitted to the returns  

---

### 4. VaR Back-testing

The VaR series are checked with the Kupiec proportion-of-failures test and the Christoffersen independence test to see whether the observed violation rates are consistent with the nominal confidence levels.

---

### 5. European Option Pricing

A short side study uses SBI stock data. Geometric-Brownian-motion paths are simulated

\[
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
\]

and the resulting terminal prices are used to price European calls and puts. Finite-difference Greeks (delta, gamma, vega, theta) are computed for a grid of strikes and maturities.

---

### 6. Delta-Neutral Hedging

Finally a simple delta-neutral overlay is constructed. Assumed deltas are:

| Asset        | Delta |
|--------------|------:|
| HDFC AMC     |   1   |
| NIFTY 50     |   1   |
| EUR/INR      |  –1   |
| Macpower CNC |   1   |
| Gold         |  0.5  |

Weights are re-solved each period so that

\[
\sum_i w_i\Delta_i = 0, \qquad \sum_i w_i = 1,
\]

with the additional constraint \(0\le w_i\le 1\) (no short sales). The resulting P&L path is compared with the original equal-weighted “buy-and-hold” portfolio.

---

### Libraries

`yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `pmdarima`, `arch`.

---
