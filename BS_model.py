"""
BS_model.py
-----------
Black-Scholes pricing model class.

Mirrors the interface of RF_model.py so both models are driven by the
same main script and results are directly comparable.

The model:
  1. Estimates σ from a trailing rolling-window of log-returns computed
     on the FULL data series (no look-ahead bias, and no NaN loss at the
     test boundary caused by recomputing vol on the test slice alone).
  2. Prices an ATM European call (K = S₀) with T = horizon trading days
     to expiry using the closed-form Black-Scholes formula.
  3. Compares that price against the discounted realised payoff
     e^{-rT} · max(S_T − K, 0).

"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


# ── module-level helpers ──────────────────────────────────────────────────────

def _bs_call_vectorised(S: np.ndarray, sigma: np.ndarray,
                        T: float, r: float) -> np.ndarray:
    """
    Vectorised Black-Scholes ATM call price (K = S₀ every row).

    Because K = S everywhere, ln(S/K) = 0, which simplifies d1/d2:
        d1 = (r + 0.5*σ²)*T / (σ*√T)  =  (r/σ + 0.5*σ)*√T
        d2 = d1 − σ*√T
    """
    sqrtT = np.sqrt(T)
    d1 = (r / sigma + 0.5 * sigma) * sqrtT
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - np.exp(-r * T) * S * norm.cdf(d2)


def _build_all_features(data: pd.DataFrame, vol_window: int,
                        horizon: int, r: float) -> pd.DataFrame:
    """
    Compute features and targets on a single sorted DataFrame.

    Rolling vol is computed on the FULL series passed in, so the caller
    is responsible for passing the full series (train + test together) and
    then slicing out the test rows after this function returns.  This
    avoids the NaN-at-boundary problem that occurs when vol is recomputed
    on the test slice alone.

    Added columns
    -------------
    log_ret       : daily log return
    sigma_trail   : trailing annualised vol  (known at each anchor date)
    S_future      : closing price `horizon` trading days ahead
    bs_call_pct   : BS ATM call price / S₀  (scale-free)
    realised_pct  : e^{-rT} · max(S_T/S₀ − 1, 0)  (scale-free)
    """
    d = data.copy().sort_values("Date").reset_index(drop=True)
    T = horizon / 252.0

    # Log returns and trailing vol
    d["log_ret"]     = np.log(d["Close"] / d["Close"].shift(1))
    d["sigma_trail"] = d["log_ret"].rolling(vol_window).std() * np.sqrt(252)

    # Forward price (look-ahead — used only for target, never as a feature)
    d["S_future"] = d["Close"].shift(-horizon)

    # Vectorised BS ATM price (only where sigma is valid)
    valid = d["sigma_trail"].notna() & (d["sigma_trail"] > 0)
    d["bs_call_pct"] = np.nan
    if valid.any():
        d.loc[valid, "bs_call_pct"] = (
            _bs_call_vectorised(
                d.loc[valid, "Close"].values,
                d.loc[valid, "sigma_trail"].values,
                T, r,
            ) / d.loc[valid, "Close"].values
        )

    # Discounted realised payoff normalised by spot
    d["realised_pct"] = (
        np.exp(-r * T)
        * np.maximum(d["S_future"] / d["Close"] - 1.0, 0.0)
    )

    return d


# ── main class ────────────────────────────────────────────────────────────────

class BS_Model:
    """
    Black-Scholes ATM call pricing and backtesting model.

    Parameters
    ----------
    df : pd.DataFrame
        Full data series (train + test) with columns Date, Close.
        Pass the full series so rolling vol is computed correctly at
        every row — the caller slices out the test subset afterwards.
    vol_window : int
        Trailing vol window in trading days.  Default: 21.
    horizon : int
        Trading days to expiry for the ATM call.  Default: 21.
    r : float
        Annualised risk-free rate.  Default: 0.04.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vol_window: int = 21,
        horizon: int = 21,
        r: float = 0.04,
    ):
        self.vol_window = vol_window
        self.horizon    = horizon
        self.r          = r
        self.metrics    = None

        # Pre-compute features on the full series once
        self._full_features = _build_all_features(df, vol_window, horizon, r)

    # ── evaluation ───────────────────────────────────────────────────────────

    def test_results(self, test_dates) -> dict:
        """
        Evaluate BS pricing on the test subset.

        Parameters
        ----------
        test_dates : array-like of dates, or pd.DataFrame with a 'Date' column
            Identifies which rows of the full series belong to the test set.

        Returns
        -------
        dict with keys: mse, rmse, mae, mape, r2, rel_mse
        """
        if isinstance(test_dates, pd.DataFrame):
            test_dates = test_dates["Date"].values

        d = self._full_features[
            self._full_features["Date"].isin(test_dates)
        ].dropna(subset=["bs_call_pct", "realised_pct"])

        y_pred = d["bs_call_pct"].values
        y_true = d["realised_pct"].values

        mse     = float(mean_squared_error(y_true, y_pred))
        rmse    = float(np.sqrt(mse))
        mae     = float(mean_absolute_error(y_true, y_pred))
        r2      = float(r2_score(y_true, y_pred))
        rel_mse = float(mse / np.var(y_true)) if np.var(y_true) > 0 else np.nan

        non_zero = y_true > 1e-8
        mape = (
            float(mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero]))
            if non_zero.sum() > 0 else np.nan
        )

        self.metrics = {
            "mse":     mse,
            "rmse":    rmse,
            "mae":     mae,
            "r2":      r2,
            "mape":    mape,
            "rel_mse": rel_mse,
        }
        return self.metrics
