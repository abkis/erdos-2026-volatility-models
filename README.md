# Erdos-2026-Volatility-Models

We will evaluate and compare the predictive performance of various volatility models during periods of extreme market stress and structural breaks. Specifically, we plan to benchmark advanced frameworks—including path-dependent volatility formulations and machine learning architectures—against traditional benchmarks like the GARCH and HAR models to determine which best captures tail-risk behavior.

## Models

We implement and compare four classes of volatility models across six large-cap equities (VGT, AAPL, MSFT, NVDA, VOO, GME) evaluated over a stable pre-COVID period and an unstable period spanning COVID-era market stress.

| Model | Description |
|-------|-------------|
| **Black-Scholes (BS)** | Constant-volatility baseline using annualized historical sigma from training returns |
| **GARCH(1,1)** | Captures volatility clustering via autoregressive conditional heteroskedasticity |
| **HAR-GK** | Heterogeneous Autoregressive model using Garman-Klass range-based estimator |
| **ML (Random Forest)** | Machine learning model trained on a rich feature set including realized volatility, EWMA, VIX proxy, and technical indicators |

## Evaluation

Models are assessed on held-out test windows using two complementary metrics:

- **MSE** – Mean Squared Error against realized forward volatility  
- **QLIKE** – Quasi-likelihood loss, less sensitive to outlier spikes and favored in the volatility forecasting literature

## Conclusion


