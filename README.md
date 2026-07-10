# Black-Scholes Model

## Overview

Implements Black-Scholes as the constant-volatility baseline for our volatility forecasting comparison. It computes historical sigma from the training window and repeats it across all test-period trading days.

## Model

Volatility is estimated as the annualized standard deviation of log-returns over the training period, then held constant as the forecast:

σ^=252⋅std ⁣(log⁡StSt−1)\hat{\sigma} = \sqrt{252} \cdot \text{std}\!\left(\log\frac{S_t}{S_{t-1}}\right)σ^=252​⋅std(logSt−1​St​​)
European option prices are then given by the Black-Scholes formula:

C=S0 Φ(d1)−Ke−rTΦ(d2),d1=ln⁡(S0/K)+(r+12σ^2)Tσ^T,d2=d1−σ^TC = S_0\,\Phi(d_1) - Ke^{-rT}\Phi(d_2), \qquad d_1 = \frac{\ln(S_0/K)+(r+\frac{1}{2}\hat{\sigma}^2)T}{\hat{\sigma}\sqrt{T}}, \qquad d_2 = d_1 - \hat{\sigma}\sqrt{T}C=S0​Φ(d1​)−Ke−rTΦ(d2​),d1​=σ^T​ln(S0​/K)+(r+21​σ^2)T​,d2​=d1​−σ^T

## Methods

MethodDescriptionvolatility_fit()Annualized historical σ^\hat{\sigma}
σ^ from training datatrading_days_test()Holiday-aware test-period trading daystest()Constant σ^\hat{\sigma}
σ^ repeated across the test windowvolatility_avg()Mean of predicted volatilities

## Notes

Black-Scholes is not designed to compete as a dynamic forecaster — it serves as the baseline every other model must beat. All dynamic models (GARCH, HAR-GK, Path Dependent, ML) are evaluated against it using MSE and QLIKE.​
