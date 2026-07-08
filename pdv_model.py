"""
Path-dependent volatility model -- Guyon & Lekeufack (2023), Section 3.2.

    vol_t = beta0 + beta1 * R1_t + beta2 * Sigma_t

R1   = kernel-weighted sum of past daily returns        (trend / leverage effect)
Sigma= sqrt(R2), R2 = kernel-weighted sum of squared returns  (volatility clustering)
Both kernels are time-shifted power laws  K(tau) = (tau + delta)^(-alpha).

Harness entry point: the `fit` class. It is handed TRAINING data only plus a test
window and must forecast forward, matching the other models in the comparison:
    fit(train_data, start_date_predict, end_date_predict).test()
returns one predicted volatility per trading day in the test window.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

BUSINESS_DAYS_PER_YEAR = 252
DT = 1.0 / BUSINESS_DAYS_PER_YEAR
_US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


# --- data extraction --------------------------------------------------------
def close_prices(data) -> pd.Series:
    """Pull the close-price Series (with its DatetimeIndex) from yfinance data.

    Handles a plain 'Close' column, a MultiIndex like ('Close', ticker), or a
    Series that is already close prices.
    """
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data.columns, pd.MultiIndex):
        for level in (0, 1):
            if "Close" in data.columns.get_level_values(level):
                c = data.xs("Close", axis=1, level=level)
                break
        else:
            raise KeyError("No 'Close' field found in the MultiIndex columns.")
    else:
        c = data["Close"]
    if isinstance(c, pd.DataFrame):        # single-ticker frame -> take its column
        c = c.iloc[:, 0]
    return c.astype(float)


def daily_returns(prices) -> np.ndarray:
    """Close-to-close simple returns from a price array/Series."""
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


def n_trading_days(start, end) -> int:
    """Number of trading days in the half-open window [start, end) (US calendar,
    matching yfinance's end-exclusive convention)."""
    rng = pd.date_range(pd.Timestamp(start),
                        pd.Timestamp(end) - pd.Timedelta(days=1), freq=_US_BDAY)
    return len(rng)


# --- kernel -----------------------------------------------------------------
def tspl_kernel(window: int, alpha: float, delta: float, dt: float = DT) -> np.ndarray:
    """Normalized time-shifted power-law weights (lag 0 = most recent), sum(w)*dt = 1."""
    if alpha <= 1.0 or delta <= 0.0:
        raise ValueError("Need alpha > 1 and delta > 0 for a valid TSPL kernel.")
    raw = (np.arange(window) * dt + delta) ** (-alpha)
    return raw / (raw.sum() * dt)


def _trailing_weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Causal weighted history: out[t] = sum_j weights[j] * values[t-j] (NaN warm-up)."""
    values = np.asarray(values, dtype=float)
    n, W = values.shape[0], weights.shape[0]
    out = np.full(n, np.nan)
    if n >= W:
        out[W - 1:] = sliding_window_view(values, W) @ weights[::-1]
    return out


# --- features ---------------------------------------------------------------
def compute_features(prices, alpha1, delta1, alpha2, delta2, window=1000, dt=DT):
    """Turn a price series into the two model inputs (R1, Sigma), aligned to prices."""
    r = daily_returns(prices)
    R1 = _trailing_weighted_sum(r, tspl_kernel(window, alpha1, delta1, dt))
    R2 = _trailing_weighted_sum(r ** 2, tspl_kernel(window, alpha2, delta2, dt))
    # returns are one shorter than prices (row-t return ends on row t); pad to align.
    R1 = np.concatenate([[np.nan], R1])
    Sigma = np.concatenate([[np.nan], np.sqrt(R2)])
    return R1, Sigma


# --- targets ----------------------------------------------------------------
def trailing_realized_vol(prices, window_days: int = 22, annualize: bool = True) -> np.ndarray:
    """Trailing annualized realized vol -- matches the project's rolling_volatility
    benchmark: pct_change().rolling(window_days).std() * sqrt(252)."""
    vol = pd.Series(np.asarray(prices, float)).pct_change().rolling(window_days).std()
    if annualize:
        vol = vol * np.sqrt(BUSINESS_DAYS_PER_YEAR)
    return vol.to_numpy()


# --- model + scoring --------------------------------------------------------
def pdv_volatility(R1, Sigma, beta0, beta1, beta2) -> np.ndarray:
    """The linear PDV model: beta0 + beta1*R1 + beta2*Sigma."""
    return beta0 + beta1 * np.asarray(R1) + beta2 * np.asarray(Sigma)


def r2_score(y_true, y_pred) -> float:
    """r^2 over finite entries."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - np.sum((yt - y_pred[m]) ** 2) / ss_tot if ss_tot > 0 else np.nan


def rmse(y_true, y_pred) -> float:
    """Root mean squared error over finite entries (volatility units)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


# --- fitting ----------------------------------------------------------------
@dataclass
class PDVParams:
    """Fitted constants of the linear PDV model, plus settings and train scores."""
    beta0: float; beta1: float; beta2: float
    alpha1: float; delta1: float; alpha2: float; delta2: float
    feature_window: int; dt: float
    train_r2: float; train_rmse: float; success: bool; message: str


_THETA0 = np.array([0.05, -0.08, 0.85, 1.50, 0.03, 1.70, 0.04])
_LOWER = np.array([-np.inf, -np.inf, -np.inf, 1.001, 1e-4, 1.001, 1e-4])
_UPPER = np.array([np.inf, np.inf, np.inf, 20.0, 2.0, 20.0, 2.0])


def learn_params(train_prices, target, feature_window: int = 1000, dt: float = DT,
                 theta0: Optional[Sequence[float]] = None, max_nfev: int = 2000) -> PDVParams:
    """Jointly fit the 7 PDV constants on training prices vs. a target vol series
    (least squares, eq. 3.7). `target` must be aligned to `train_prices`."""
    prices = np.asarray(train_prices, float)
    target = np.asarray(target, float)
    n = len(prices)
    if n <= feature_window + 1:
        raise ValueError(f"Need more than feature_window+1={feature_window+1} prices; got {n}.")

    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[feature_window:] = True                 # first valid feature row
    fit_mask &= np.isfinite(target)
    if fit_mask.sum() < 10:
        raise ValueError("Too few usable rows to fit; supply more training history.")

    def residuals(theta):
        b0, b1, b2, a1, d1, a2, d2 = theta
        R1, Sigma = compute_features(prices, a1, d1, a2, d2, feature_window, dt)
        return (pdv_volatility(R1, Sigma, b0, b1, b2) - target)[fit_mask]

    res = least_squares(residuals, _THETA0 if theta0 is None else np.asarray(theta0, float),
                        bounds=(_LOWER, _UPPER), method="trf", x_scale="jac", max_nfev=max_nfev)

    b0, b1, b2, a1, d1, a2, d2 = res.x
    R1, Sigma = compute_features(prices, a1, d1, a2, d2, feature_window, dt)
    pred = pdv_volatility(R1, Sigma, b0, b1, b2)
    return PDVParams(
        beta0=b0, beta1=b1, beta2=b2, alpha1=a1, delta1=d1, alpha2=a2, delta2=d2,
        feature_window=feature_window, dt=dt,
        train_r2=r2_score(target[fit_mask], pred[fit_mask]),
        train_rmse=rmse(target[fit_mask], pred[fit_mask]),
        success=bool(res.success), message=str(res.message),
    )


# --- prediction / forecasting -----------------------------------------------
def predict_volatility(params: PDVParams, prices) -> np.ndarray:
    """In-sample (contemporaneous) predicted vol from a price series; needs future
    returns to be known, so it's for analysis, not out-of-sample forecasting."""
    R1, Sigma = compute_features(prices, params.alpha1, params.delta1,
                                 params.alpha2, params.delta2, params.feature_window, params.dt)
    return pdv_volatility(R1, Sigma, params.beta0, params.beta1, params.beta2)


def forecast_volatility(params: PDVParams, train_prices, horizon: int) -> np.ndarray:
    """Multi-step-ahead forecast for `horizon` trading days past the training end,
    using only past returns.

    The two features are rolled forward along their expected path: each future day
    contributes expected return 0 to R1 (the trend impulse fades) and expected
    squared return sigma^2/252 to R2 (so Sigma mean-reverts toward the forecast
    level). This produces a smooth mean-reverting forecast curve.
    """
    r = daily_returns(train_prices)
    W = params.feature_window
    w1r = tspl_kernel(W, params.alpha1, params.delta1, params.dt)[::-1]
    w2r = tspl_kernel(W, params.alpha2, params.delta2, params.dt)[::-1]

    buf_r = r[-W:].astype(float).copy()          # trailing returns (oldest -> newest)
    buf_r2 = buf_r ** 2                           # trailing squared returns
    out = np.empty(horizon)
    for h in range(horizon):
        R1 = buf_r @ w1r
        R2 = buf_r2 @ w2r
        sig = params.beta0 + params.beta1 * R1 + params.beta2 * np.sqrt(max(R2, 0.0))
        sig = max(sig, 1e-8)                       # keep positive (QLIKE-safe)
        out[h] = sig
        buf_r = np.append(buf_r[1:], 0.0)          # E[next return] = 0
        buf_r2 = np.append(buf_r2[1:], sig * sig * DT)   # E[next return^2] = sig^2/252
    return out


# --- harness interface ------------------------------------------------------
class fit:
    """PDV model for the comparison harness.

    Parameters
    ----------
    data : yfinance TRAINING DataFrame for one ticker (as produced by
        dowloand_data.training_data()); the test window is forecast, not read
        from here.
    start_date_predict, end_date_predict : test window; its US trading days set
        the number of forecast steps (half-open [start, end), like yfinance).
    feature_window : PDV kernel look-back (paper uses 1000; auto-shrunk for short
        training histories so ~half the data is left for fitting).
    target_window : realized-vol window used as the training target; 22 matches
        the project's rolling_volatility benchmark.
    """

    def __init__(self, data, start_date_predict, end_date_predict,
                 feature_window: int = 1000, target_window: int = 22):
        prices = close_prices(data)
        self.train_prices = prices.to_numpy(dtype=float)
        n = self.train_prices.size
        self.horizon = n_trading_days(start_date_predict, end_date_predict)

        # short training sets can't support a 1000-day kernel; keep enough fit rows
        self.feature_window = int(min(feature_window, max(60, n // 2)))
        target = trailing_realized_vol(self.train_prices, target_window)
        self.params = learn_params(self.train_prices, target, feature_window=self.feature_window)

    def test(self) -> np.ndarray:
        """Predicted volatility for each trading day in the test window."""
        return forecast_volatility(self.params, self.train_prices, self.horizon)
