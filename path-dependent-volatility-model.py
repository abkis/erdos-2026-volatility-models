"""
Path-dependent volatility model -- Guyon & Lekeufack (2023), Section 3.2.

    vol_t = beta0 + beta1 * R1_t + beta2 * Sigma_t

R1   = kernel-weighted sum of past daily returns        (trend / leverage effect)
Sigma= sqrt(R2), R2 = kernel-weighted sum of squared returns  (volatility clustering)
Both kernels are time-shifted power laws  K(tau) = (tau + delta)^(-alpha).

Inputs are price DataFrames with a "Close" column; close-to-close returns are the
only thing the model uses. The target is forward realized vol over `horizon` days.
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


# --- data -------------------------------------------------------------------
def daily_returns(df: pd.DataFrame) -> np.ndarray:
    """Close-to-close simple returns from a price DataFrame."""
    if "Close" not in df:
        raise KeyError('DataFrame must have a "Close" column.')
    close = np.asarray(df["Close"], dtype=float)
    return close[1:] / close[:-1] - 1.0


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
def compute_features(df, alpha1, delta1, alpha2, delta2, window=1000, dt=DT):
    """Turn price data into the two model inputs (R1, Sigma), aligned to df rows."""
    r = daily_returns(df)
    R1 = _trailing_weighted_sum(r, tspl_kernel(window, alpha1, delta1, dt))
    R2 = _trailing_weighted_sum(r ** 2, tspl_kernel(window, alpha2, delta2, dt))
    # returns are one shorter than prices (row-t return ends on row t); pad to align.
    R1 = np.concatenate([[np.nan], R1])
    Sigma = np.concatenate([[np.nan], np.sqrt(R2)])
    return R1, Sigma


# --- target -----------------------------------------------------------------
def forward_realized_vol(df, horizon: int = 1, annualize: bool = True) -> np.ndarray:
    """Annualized realized vol over the next `horizon` trading days (model target).

    Aligned to df rows: value at t covers days t+1..t+horizon. Small horizons are
    noisier (a 1-day estimate from one daily return has high sampling noise);
    ~5-10 days is a good balance for daily data.
    """
    ret = np.concatenate([[np.nan], daily_returns(df)])     # row-t return
    fwd_sumsq = pd.Series(ret ** 2).rolling(horizon).sum().shift(-horizon)
    scale = (BUSINESS_DAYS_PER_YEAR / horizon) if annualize else (1.0 / horizon)
    return np.sqrt(scale * fwd_sumsq.to_numpy())


# --- model + scoring --------------------------------------------------------
def pdv_volatility(R1, Sigma, beta0, beta1, beta2) -> np.ndarray:
    """The linear PDV model: beta0 + beta1*R1 + beta2*Sigma."""
    return beta0 + beta1 * np.asarray(R1) + beta2 * np.asarray(Sigma)


def r2_score(y_true, y_pred) -> float:
    """r^2 over finite entries (the score reported in the paper)."""
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
    horizon: int; window: int; dt: float
    train_r2: float; train_rmse: float; success: bool; message: str


_THETA0 = np.array([0.05, -0.08, 0.85, 1.50, 0.03, 1.70, 0.04])
_LOWER = np.array([-np.inf, -np.inf, -np.inf, 1.001, 1e-4, 1.001, 1e-4])
_UPPER = np.array([np.inf, np.inf, np.inf, 20.0, 2.0, 20.0, 2.0])


def learn_params(train_df, horizon: int = 1, window: int = 1000, dt: float = DT,
                 theta0: Optional[Sequence[float]] = None, max_nfev: int = 2000) -> PDVParams:
    """Jointly fit the 7 PDV constants on training prices by least squares (eq. 3.7).

    The kernel parameters live inside the features, so R1/Sigma are rebuilt on
    every step -- this is a non-convex fit, and `theta0` is the starting point.
    """
    target = forward_realized_vol(train_df, horizon)
    n = len(train_df)
    if n <= window:
        raise ValueError(f"Need more than window={window} rows; got {n}.")

    fit_mask = np.zeros(n, dtype=bool)
    fit_mask[window:] = True                 # first valid feature row is `window`
    fit_mask &= np.isfinite(target)
    if fit_mask.sum() < 10:
        raise ValueError("Too few usable rows to fit.")

    def residuals(theta):
        b0, b1, b2, a1, d1, a2, d2 = theta
        R1, Sigma = compute_features(train_df, a1, d1, a2, d2, window, dt)
        return (pdv_volatility(R1, Sigma, b0, b1, b2) - target)[fit_mask]

    res = least_squares(residuals, _THETA0 if theta0 is None else np.asarray(theta0, float),
                        bounds=(_LOWER, _UPPER), method="trf", x_scale="jac", max_nfev=max_nfev)

    b0, b1, b2, a1, d1, a2, d2 = res.x
    R1, Sigma = compute_features(train_df, a1, d1, a2, d2, window, dt)
    pred = pdv_volatility(R1, Sigma, b0, b1, b2)
    return PDVParams(
        beta0=b0, beta1=b1, beta2=b2, alpha1=a1, delta1=d1, alpha2=a2, delta2=d2,
        horizon=horizon, window=window, dt=dt,
        train_r2=r2_score(target[fit_mask], pred[fit_mask]),
        train_rmse=rmse(target[fit_mask], pred[fit_mask]),
        success=bool(res.success), message=str(res.message),
    )


# --- prediction + evaluation ------------------------------------------------
def predict_volatility(params: PDVParams, df) -> np.ndarray:
    """Predicted forward vol from prices. Value at row t forecasts days t+1..t+horizon."""
    R1, Sigma = compute_features(df, params.alpha1, params.delta1,
                                 params.alpha2, params.delta2, params.window, params.dt)
    return pdv_volatility(R1, Sigma, params.beta0, params.beta1, params.beta2)


def evaluate(params: PDVParams, df) -> dict:
    """Out-of-sample r^2 and RMSE against realized forward vol on a price DataFrame."""
    pred = predict_volatility(params, df)
    actual = forward_realized_vol(df, params.horizon)
    return {"r2": r2_score(actual, pred), "rmse": rmse(actual, pred)}


# --- optional plotting ------------------------------------------------------
def plot_fit(params: PDVParams, df, dates=None, title="PDV fit"):
    """Diagnostic plots: predicted vs realized vol through time, and a scatter."""
    import matplotlib.pyplot as plt
    pred = predict_volatility(params, df)
    actual = forward_realized_vol(df, params.horizon)
    m = np.isfinite(pred) & np.isfinite(actual)
    x = (np.asarray(dates) if dates is not None else np.arange(len(pred)))[m]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(x, actual[m], lw=1, label="realized")
    ax[0].plot(x, pred[m], lw=1, label="predicted")
    ax[0].set_title(f"{title}  (r2={r2_score(actual, pred):.3f})"); ax[0].legend()
    lo, hi = actual[m].min(), actual[m].max()
    ax[1].scatter(actual[m], pred[m], s=6, alpha=0.4)
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[1].set_xlabel("realized vol"); ax[1].set_ylabel("predicted vol")
    fig.tight_layout()
    return fig
