"""
ML_BS_model.py
--------------
Random-Forest model that learns to CORRECT Black-Scholes mispricing.

Workflow
--------
1. Compute normalised BS ATM call price (price / S₀) and all features on
   the FULL data series (train + test) in one pass so rolling-vol windows
   never lose rows at the train/test boundary.
2. Train the RF on the training-date rows only, targeting the BS residual:
       residual = realised_pct − bs_call_pct
3. At test time: final prediction = bs_call_pct + RF_correction.
4. Report the same metric set as BS_model.py (MSE, RMSE, MAE, MAPE, R², rel_MSE).

All prices are normalised by S₀ so metrics are scale-free and directly
comparable with BS_model.py and RF_model.py.

Features (all known at the anchor date — zero look-ahead)
---------------------------------------------------------
sigma_trail    trailing annualised vol (short window)
log_ret_1d     1-day log-return
log_ret_5d     5-day log-return
log_ret_21d    21-day log-return
bs_call_pct    normalised BS price (RF corrects the residual from it)
bs_delta       Δ  — option delta
bs_gamma_norm  Γ · S₀  (dimensionless)
bs_vega_norm   ν / S₀  (dimensionless)
high_low_ratio H / L intraday range proxy
vol_ratio      σ_short / σ_long  (vol-of-vol signal)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


# ── module-level BS helpers (vectorised where possible) ───────────────────────

def _d1(S, K, sigma, T, r):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _bs_call_vec(S, sigma, T, r):
    """Vectorised ATM BS call price (K = S everywhere)."""
    sqrtT = np.sqrt(T)
    d1 = (r / sigma + 0.5 * sigma) * sqrtT
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - np.exp(-r * T) * S * norm.cdf(d2)


def _bs_delta_vec(sigma, T, r):
    """ATM call delta (K = S, so ln(S/K)=0)."""
    d1 = (r / sigma + 0.5 * sigma) * np.sqrt(T)
    return norm.cdf(d1)


def _bs_gamma_vec(S, sigma, T):
    """ATM gamma × S₀ (dimensionless)."""
    d1 = (np.sqrt(T) * (sigma**2) / 2 + np.log(1.0)) / (sigma * np.sqrt(T))
    # For ATM: d1 = 0.5*sigma*sqrt(T) (when r≈0 for gamma, or full form)
    # Full form: use precomputed d1 from delta calc
    return norm.pdf(d1) / (sigma * np.sqrt(T))   # gamma * S / S = gamma * S normalised


def _bs_vega_vec(S, sigma, T):
    """ATM vega / S₀ (dimensionless)."""
    d1 = 0.5 * sigma * np.sqrt(T)   # ATM approximation with r≈0 for d1
    return np.sqrt(T) * norm.pdf(d1)  # vega/S


def _build_all_features(data: pd.DataFrame, vol_window: int,
                        horizon: int, r: float) -> pd.DataFrame:
    """
    Compute all ML features and targets on the full sorted DataFrame.

    Forward-looking columns (S_future, realised_pct, bs_residual) are
    present but are NEVER used as model features — only as training targets.
    """
    d = data.copy().sort_values("Date").reset_index(drop=True)
    T = horizon / 252.0

    log_ret = np.log(d["Close"] / d["Close"].shift(1))

    d["sigma_trail"]    = log_ret.rolling(vol_window).std()     * np.sqrt(252)
    d["sigma_long"]     = log_ret.rolling(vol_window * 3).std() * np.sqrt(252)
    d["log_ret_1d"]     = log_ret
    d["log_ret_5d"]     = np.log(d["Close"] / d["Close"].shift(5))
    d["log_ret_21d"]    = np.log(d["Close"] / d["Close"].shift(21))
    d["high_low_ratio"] = d["High"] / d["Low"]
    d["vol_ratio"]      = d["sigma_trail"] / d["sigma_long"]

    # Vectorised BS price and Greeks (ATM: K = S₀)
    valid = d["sigma_trail"].notna() & (d["sigma_trail"] > 0)
    d["bs_call_pct"]   = np.nan
    d["bs_delta"]      = np.nan
    d["bs_gamma_norm"] = np.nan
    d["bs_vega_norm"]  = np.nan

    if valid.any():
        S_v  = d.loc[valid, "Close"].values
        sig_v = d.loc[valid, "sigma_trail"].values
        sqrtT = np.sqrt(T)
        d1v = (r / sig_v + 0.5 * sig_v) * sqrtT
        d2v = d1v - sig_v * sqrtT
        call_v = S_v * norm.cdf(d1v) - np.exp(-r * T) * S_v * norm.cdf(d2v)

        d.loc[valid, "bs_call_pct"]   = call_v / S_v
        d.loc[valid, "bs_delta"]      = norm.cdf(d1v)
        d.loc[valid, "bs_gamma_norm"] = norm.pdf(d1v) / (sig_v * sqrtT)  # Γ·S₀/S₀ = Γ (dimensionless)
        d.loc[valid, "bs_vega_norm"]  = sqrtT * norm.pdf(d1v)             # ν/S₀

    # Forward-looking — target construction only, never a feature
    d["S_future"]    = d["Close"].shift(-horizon)
    d["realised_pct"] = (
        np.exp(-r * T)
        * np.maximum(d["S_future"] / d["Close"] - 1.0, 0.0)
    )
    d["bs_residual"] = d["realised_pct"] - d["bs_call_pct"]

    return d


# ── main class ────────────────────────────────────────────────────────────────

class ML_BS_Model:
    """
    Random-Forest residual corrector for Black-Scholes ATM call prices.

    Parameters
    ----------
    df : pd.DataFrame
        Full data series (train + test) with columns Date, Close, High, Low.
        Features are computed on the full series so rolling windows are
        valid at every row.
    vol_window : int
        Short trailing-vol window (trading days).  Default: 21.
    horizon : int
        Days to expiry for the ATM call.  Default: 21.
    r : float
        Annualised risk-free rate.  Default: 0.04.
    rf_params : dict | None
        Hyperparameters for RandomForestRegressor.
    """

    FEATURE_COLS = [
        "sigma_trail",
        "log_ret_1d",
        "log_ret_5d",
        "log_ret_21d",
        "bs_call_pct",
        "bs_delta",
        "bs_gamma_norm",
        "bs_vega_norm",
        "high_low_ratio",
        "vol_ratio",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        vol_window: int = 21,
        horizon: int = 21,
        r: float = 0.04,
        rf_params: dict | None = None,
    ):
        self.vol_window = vol_window
        self.horizon    = horizon
        self.r          = r
        self.rf         = None
        self.metrics    = None

        self.rf_params = rf_params or {
            "n_estimators":     500,
            "max_depth":        12,
            "min_samples_leaf":  5,
            "max_features":     0.5,
            "random_state":     42,
            "n_jobs":          -1,
        }

        # Compute features on the full series once
        self._full_features = _build_all_features(df, vol_window, horizon, r)

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, train_dates) -> None:
        """
        Train the RF on the training-date rows only.

        Parameters
        ----------
        train_dates : array-like of dates, or pd.DataFrame with a 'Date' column
        """
        if isinstance(train_dates, pd.DataFrame):
            train_dates = train_dates["Date"].values

        d = self._full_features[
            self._full_features["Date"].isin(train_dates)
        ].dropna(subset=self.FEATURE_COLS + ["bs_residual"])

        X = d[self.FEATURE_COLS].values
        y = d["bs_residual"].values

        self.rf = RandomForestRegressor(**self.rf_params)
        self.rf.fit(X, y)

    # ── evaluation ────────────────────────────────────────────────────────────

    def test_results(self, test_dates) -> dict:
        """
        Evaluate ML-corrected BS on test-date rows.

        Parameters
        ----------
        test_dates : array-like of dates, or pd.DataFrame with a 'Date' column

        Returns
        -------
        dict with keys: mse, rmse, mae, mape, r2, rel_mse
        """
        if self.rf is None:
            raise RuntimeError("Call fit() before test_results().")

        if isinstance(test_dates, pd.DataFrame):
            test_dates = test_dates["Date"].values

        d = self._full_features[
            self._full_features["Date"].isin(test_dates)
        ].dropna(subset=self.FEATURE_COLS + ["realised_pct"])

        correction = self.rf.predict(d[self.FEATURE_COLS].values)
        y_pred = d["bs_call_pct"].values + correction
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

    def feature_importance(self) -> pd.Series:
        """Return RF feature importances sorted descending."""
        if self.rf is None:
            raise RuntimeError("Call fit() first.")
        return (
            pd.Series(self.rf.feature_importances_, index=self.FEATURE_COLS)
            .sort_values(ascending=False)
        )
