# Black-Scholes Model

## Overview

Implements Black-Scholes as the constant-volatility baseline for our volatility forecasting comparison. It computes historical sigma from the training window and repeats it across all test-period trading days.

## Overview

Implements Black-Scholes as the **constant-volatility baseline** for our volatility forecasting comparison. Historical sigma is estimated from training returns and held fixed across the entire test window.

## Model

Volatility is estimated as the annualized standard deviation of log-returns over the training period, then held constant as the forecast:

​```
σ = sqrt(252) * std(log(S_t / S_{t-1}))
​```

European option prices are then given by the Black-Scholes formula:

​```
d1 = (log(S0/K) + (r + 0.5*σ²)*T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
C  = S0 * Φ(d1) - K * exp(-r*T) * Φ(d2)
​```

## Methods

| Method | Description |
|---|---|
| `volatility_fit()` | Annualized historical `σ` from training data |
| `trading_days_test()` | Holiday-aware test-period trading days |
| `test()` | Constant `σ` repeated across the test window |
| `volatility_avg()` | Mean of predicted volatilities |

## Notes

Black-Scholes is not designed to compete as a dynamic forecaster — it serves as the **baseline** every other model must beat. All dynamic models (GARCH, HAR-GK, Path Dependent, ML) are evaluated against it using MSE and QLIKE.


