"""
Path-dependent volatility model -- Guyon & Lekeufack (2023), Section 3.2.

    vol_t = beta0 + beta1 * R1_t + beta2 * Sigma_t

R1   = kernel-weighted sum of past daily returns        (trend / leverage effect)
Sigma= sqrt(R2), R2 = kernel-weighted sum of squared returns  (volatility clustering)
Both kernels are time-shifted power laws  K(tau) = (tau + delta)^(-alpha).

Harness entry point: the `fit` class. It is given the FULL price history plus a
test window. Only the pre-test portion is used to learn the seven constants; the
predictions then use the ACTUAL returns available up to each test day (training
returns plus the test returns that precede it):

    fit(data, start_date_predict, end_date_predict).test()

`data` must therefore span through end_date_predict (e.g. training + test data).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares

BUSINESS_DAYS_PER_YEAR = 252
DT = 1.0 / BUSINESS_DAYS_PER_YEAR


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
    """Turn a price series into the two model inputs (R1, Sigma), aligned to prices.
    Feature at row t uses the returns realized up to and including day t."""
    r = daily_returns(prices)
    R1 = _trailing_weighted_sum(r, tspl_kernel(window, alpha1, delta1, dt))
    R2 = _trailing_weighted_sum(r ** 2, tspl_kernel(window, alpha2, delta2, dt))
    # returns are one shorter than prices (row-t return ends on row t); pad to align.
    R1 = np.concatenate([[np.nan], R1])
    Sigma = np.concatenate([[np.nan], np.sqrt(R2)])
    return R1, Sigma


# --- target -----------------------------------------------------------------
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


def predict_volatility(params: PDVParams, prices) -> np.ndarray:
    """Predicted volatility across a price series, aligned to its rows (row t uses
    the actual returns up to and including day t)."""
    R1, Sigma = compute_features(prices, params.alpha1, params.delta1,
                                 params.alpha2, params.delta2, params.feature_window, params.dt)
    return pdv_volatility(R1, Sigma, params.beta0, params.beta1, params.beta2)


# --- harness interface ------------------------------------------------------
class fit:
    """PDV model for the comparison harness.

    Parameters
    ----------
    data : yfinance DataFrame for one ticker spanning the FULL history through
        end_date_predict (e.g. pd.concat([df.training_data(), df.test_data()])).
    start_date_predict, end_date_predict : test window [start, end) (yfinance
        end-exclusive). Parameters are learned only from rows before start;
        predictions use the actual returns available up to each test day.
    feature_window : PDV kernel look-back (paper uses 1000; auto-shrunk for short
        training histories so ~half the training rows are left for fitting).
    target_window : realized-vol window used as the training target; 22 matches
        the project's rolling_volatility benchmark.
    """

    def __init__(self, data, start_date_predict, end_date_predict,
                 feature_window: int = 1000, target_window: int = 22):
        prices = close_prices(data)
        self.dates = prices.index
        self.prices = prices.to_numpy(dtype=float)

        start = pd.Timestamp(start_date_predict)
        end = pd.Timestamp(end_date_predict)
        self.test_mask = (self.dates >= start) & (self.dates < end)
        train_mask = self.dates < start

        if self.test_mask.sum() == 0:
            raise ValueError(
                "No rows in [start_date_predict, end_date_predict) were found in `data`. "
                "This model needs data spanning the test window -- pass the full history "
                "through end_date_predict, e.g. pd.concat([df.training_data(), df.test_data()]), "
                "not just training_data().")

        train_prices = self.prices[train_mask]
        n_train = train_prices.size
        if n_train < 40:
            raise ValueError("Too little training history before start_date_predict.")

        self.feature_window = int(min(feature_window, max(60, n_train // 2)))
        target = trailing_realized_vol(train_prices, target_window)
        self.params = learn_params(train_prices, target, feature_window=self.feature_window)

    def test(self) -> np.ndarray:
        """Predicted volatility for each test day, using the actual returns up to
        (but not including) that day. One value per trading day in the window."""
        pred = predict_volatility(self.params, self.prices)   # over the FULL series
        rows = np.flatnonzero(self.test_mask)
        out = pred[rows - 1]                                  # returns prior to each test day
        return np.clip(out, 1e-8, None)                       # keep positive for QLIKE
