# Black-Scholes Model

## Overview

Implements Black-Scholes as the **constant-volatility baseline** for our volatility forecasting comparison. Historical sigma is estimated from training returns and held fixed across the entire test window.

## Model

Volatility is estimated as the annualized standard deviation of log-returns over the training period, then held constant as the forecast:

$$\hat{\sigma} = \sqrt{252} \cdot \text{std}\left(\log\frac{S_t}{S_{t-1}}\right)$$

European option prices are then given by the Black-Scholes formula:

$$d_1 = \frac{\ln(S_0/K) + \left(r + \frac{1}{2}\hat{\sigma}^2\right)T}{\hat{\sigma}\sqrt{T}}, \qquad d_2 = d_1 - \hat{\sigma}\sqrt{T}$$

$$C = S_0\,\Phi(d_1) - Ke^{-rT}\Phi(d_2)$$

## Methods

| Method | Description |
|---|---|
| `volatility_fit()` | Annualized historical `σ` from training data |
| `trading_days_test()` | Holiday-aware test-period trading days |
| `test()` | Constant `σ` repeated across the test window |
| `volatility_avg()` | Mean of predicted volatilities |

## Notes

Black-Scholes is not designed to compete as a dynamic forecaster — it serves as the **baseline** every other model must beat. All dynamic models (GARCH, HAR-GK, Path Dependent, ML) are evaluated against it using MSE and QLIKE.


