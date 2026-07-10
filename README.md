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

# Individual Branches

The main branch contains the final notebook file which compares all models. However, certain branches contain additional information

## ML-model-notes

- Contains notes on papers related to machine learning models and relevant volatility markers
- Contains jupyter notebooks with explorations of use of the models
- Contains files for elastic net and random forest model implementation
- main.py can be run to test model and produces graphs of results

  ## Conclusion

- **Path Dependent** is the best overall model.
- **ML Model** Competitive but shows some inconsistencies
- **HAR-GK and GARCH** performs reasonably on stable assets but degrades sharply on
  high-volatility tickers, making it the weakest of the dynamic models overall.
- **Black-Scholes** ranks last on MSE due to its constant-volatility assumption,
  yet remains competitive under QLIKE — serving as a reliable pricing floor
  rather than a forecasting tool.
