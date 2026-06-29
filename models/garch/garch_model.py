"""GARCH(1,1) volatility model for the project pipeline.

The class takes cleaned price data for one ticker and returns one-step-ahead
volatility forecasts. It does not download data. The input dataframe should
contain at least a ``Close`` column; ``Open``, ``High``, ``Low``, and ``Volume``
can also be included by the data-cleaning step.

Example
-------
from garch_model import GARCHModel

model = GARCHModel(train_df, ticker="^GSPC")
predicted_vol = model.predict(test_df)
metrics = model.test(test_df)

Notes
-----
GARCH is estimated with percentage log returns,
``100 * log(Close_t / Close_{t-1})``. Forecasts are returned as decimal daily
volatility by default, so 0.012 means 1.2% daily volatility. Set
``annualize_output=True`` to return annualized volatility instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


@dataclass
class GARCHParams:
    """Fitted parameters for GARCH(1,1)."""

    mu: float
    omega: float
    alpha: float
    beta: float

    @property
    def alpha_plus_beta(self) -> float:
        return self.alpha + self.beta


class GARCHModel:
    """One-step-ahead GARCH(1,1) volatility forecaster.

    Parameters
    ----------
    train_df:
        Training data for one ticker. Must contain ``Close``.
    ticker:
        Optional ticker label used in output dictionaries.
    price_col:
        Price column used to compute returns.
    realized_vol_method:
        Realized-volatility proxy for ``test``. Options are ``abs_return``,
        ``squared_return``, and ``parkinson``. The Parkinson proxy requires
        ``High`` and ``Low`` columns.
    annualize_output:
        If True, predictions and test targets are multiplied by ``sqrt(252)``.
    trading_days:
        Number of trading days used for annualization.
    min_obs:
        Minimum number of returns needed to fit the model.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        ticker: Optional[str] = None,
        price_col: str = "Close",
        realized_vol_method: str = "abs_return",
        annualize_output: bool = False,
        trading_days: int = 252,
        min_obs: int = 50,
    ) -> None:
        self.ticker = ticker
        self.price_col = price_col
        self.realized_vol_method = realized_vol_method
        self.annualize_output = annualize_output
        self.trading_days = trading_days
        self.min_obs = min_obs

        self.train_df = self._prepare_dataframe(train_df)
        self.params_: Optional[GARCHParams] = None
        self.fitted_: bool = False
        self.fit_success_: bool = False
        self.fit_message_: str = "Not fitted yet."

        self.train_returns_: Optional[pd.Series] = None
        self.conditional_variance_: Optional[pd.Series] = None
        self.conditional_volatility_: Optional[pd.Series] = None

        self.last_close_: Optional[float] = None
        self.last_epsilon_: Optional[float] = None
        self.last_sigma2_: Optional[float] = None

        self.fit(self.train_df)

    def fit(self, train_df: Optional[pd.DataFrame] = None) -> "GARCHModel":
        """Estimate GARCH(1,1) parameters from the training dataframe."""
        if train_df is not None:
            self.train_df = self._prepare_dataframe(train_df)

        returns = self._compute_log_returns_percent(self.train_df)
        if len(returns) < self.min_obs:
            raise ValueError(
                f"Not enough return observations to fit GARCH: "
                f"need at least {self.min_obs}, got {len(returns)}."
            )

        self.train_returns_ = returns
        mu = float(returns.mean())
        eps = returns.to_numpy(dtype=float) - mu
        sample_var = float(np.var(eps, ddof=1))
        if not np.isfinite(sample_var) or sample_var <= 0:
            raise ValueError("Training returns have zero or invalid variance.")

        omega, alpha, beta, success, message = self._estimate_garch_params(eps, sample_var)
        self.params_ = GARCHParams(mu=mu, omega=omega, alpha=alpha, beta=beta)
        self.fit_success_ = bool(success)
        self.fit_message_ = str(message)

        sigma2 = self._garch_recursion(eps, omega, alpha, beta, initial_variance=sample_var)
        self.conditional_variance_ = pd.Series(
            sigma2,
            index=returns.index,
            name="garch_variance_percent2",
        )
        self.conditional_volatility_ = pd.Series(
            np.sqrt(np.maximum(sigma2, 1e-12)),
            index=returns.index,
            name="garch_volatility_percent",
        )

        # Store the final in-sample state so test-period forecasts start from
        # the end of the training period.
        self.last_close_ = float(self.train_df[self.price_col].iloc[-1])
        self.last_epsilon_ = float(eps[-1])
        self.last_sigma2_ = float(sigma2[-1])
        self.fitted_ = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        annualize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """Return one-step-ahead volatility forecasts for the given dataframe."""
        self._check_fitted()
        pred_df = self._prepare_dataframe(df)
        pred_df = self._filter_dates(pred_df, start_date, end_date)
        if pred_df.empty:
            return np.array([], dtype=float)

        use_annualized = self.annualize_output if annualize_output is None else annualize_output
        returns = self._compute_prediction_returns_percent(pred_df)
        pred_vol_percent = self._sequential_garch_forecast_percent(returns)
        pred_vol_decimal = pred_vol_percent / 100.0
        if use_annualized:
            pred_vol_decimal = pred_vol_decimal * np.sqrt(self.trading_days)
        return pred_vol_decimal.to_numpy(dtype=float)

    def test(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate forecasts against a realized-volatility proxy."""
        self._check_fitted()
        test_df = self._prepare_dataframe(df)
        test_df = self._filter_dates(test_df, start_date, end_date)
        if test_df.empty:
            raise ValueError("Test dataframe is empty after applying date filters.")

        y_pred = self.predict(test_df, annualize_output=self.annualize_output)
        y_true = self._realized_volatility_proxy(test_df, annualize_output=self.annualize_output)

        n = min(len(y_pred), len(y_true))
        y_pred = np.asarray(y_pred[:n], dtype=float)
        y_true = np.asarray(y_true[:n], dtype=float)
        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(y_true) == 0:
            raise ValueError("No valid observations available for testing.")

        err = y_pred - y_true
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)

        return {
            "model": "GARCH(1,1)",
            "ticker": self.ticker if self.ticker is not None else "",
            "n_obs": int(len(y_true)),
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mae": mae,
        }

    def get_fitted_params(self) -> Dict[str, float]:
        """Return fitted parameters and convergence status."""
        self._check_fitted()
        assert self.params_ is not None
        return {
            "mu": self.params_.mu,
            "omega": self.params_.omega,
            "alpha": self.params_.alpha,
            "beta": self.params_.beta,
            "alpha_plus_beta": self.params_.alpha_plus_beta,
            "fit_success": float(self.fit_success_),
        }

    def predict_series(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        annualize_output: Optional[bool] = None,
    ) -> pd.Series:
        """Same forecast as ``predict``, returned as a date-indexed Series."""
        pred_df = self._prepare_dataframe(df)
        pred_df = self._filter_dates(pred_df, start_date, end_date)
        values = self.predict(pred_df, annualize_output=annualize_output)
        return pd.Series(values, index=pred_df.index[: len(values)], name="garch_predicted_volatility")

    def _check_fitted(self) -> None:
        if not self.fitted_ or self.params_ is None:
            raise RuntimeError("GARCHModel must be fitted before prediction/testing.")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        out = df.copy()
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"])
            out = out.set_index("Date")

        if not isinstance(out.index, pd.DatetimeIndex):
            try:
                out.index = pd.to_datetime(out.index)
            except Exception:
                pass

        if self.price_col not in out.columns:
            raise ValueError(f"DataFrame must contain price column `{self.price_col}`.")

        out = out.sort_index()
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=[self.price_col])
        out = out[out[self.price_col].astype(float) > 0]
        return out

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> pd.DataFrame:
        out = df
        if isinstance(out.index, pd.DatetimeIndex):
            if start_date is not None:
                out = out.loc[out.index >= pd.to_datetime(start_date)]
            if end_date is not None:
                out = out.loc[out.index <= pd.to_datetime(end_date)]
        return out

    def _compute_log_returns_percent(self, df: pd.DataFrame) -> pd.Series:
        close = df[self.price_col].astype(float)
        returns = 100.0 * np.log(close / close.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        returns.name = "log_return_percent"
        return returns

    def _compute_prediction_returns_percent(self, df: pd.DataFrame) -> pd.Series:
        assert self.last_close_ is not None
        close = df[self.price_col].astype(float)
        values = []
        prev_close = float(self.last_close_)

        for current_close in close.to_numpy(dtype=float):
            values.append(100.0 * np.log(current_close / prev_close))
            prev_close = float(current_close)

        return pd.Series(values, index=df.index, name="prediction_return_percent")

    def _estimate_garch_params(
        self,
        eps: np.ndarray,
        sample_var: float,
    ) -> Tuple[float, float, float, bool, str]:
        """Estimate omega, alpha, and beta by Gaussian quasi-MLE."""
        x0 = np.array([max(sample_var * 0.05, 1e-8), 0.05, 0.90], dtype=float)
        bounds = [(1e-12, None), (1e-8, 0.999), (1e-8, 0.999)]

        def objective(x: np.ndarray) -> float:
            omega, alpha, beta = x
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
                return 1e12

            sigma2 = self._garch_recursion(eps, omega, alpha, beta, initial_variance=sample_var)
            sigma2 = np.maximum(sigma2, 1e-12)
            nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + (eps**2) / sigma2)
            return float(nll) if np.isfinite(nll) else 1e12

        if minimize is None:
            warnings.warn(
                "scipy.optimize is unavailable; using fallback GARCH parameters.",
                RuntimeWarning,
            )
            omega, alpha, beta = x0
            return float(omega), float(alpha), float(beta), False, "scipy unavailable; fallback parameters used"

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            warnings.warn(
                f"GARCH optimization did not fully converge: {result.message}. "
                "Using the best available parameter values.",
                RuntimeWarning,
            )

        omega, alpha, beta = result.x
        return float(omega), float(alpha), float(beta), bool(result.success), str(result.message)

    @staticmethod
    def _garch_recursion(
        eps: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        initial_variance: float,
    ) -> np.ndarray:
        n = len(eps)
        sigma2 = np.empty(n, dtype=float)
        sigma2[0] = max(initial_variance, 1e-12)

        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            if not np.isfinite(sigma2[t]) or sigma2[t] <= 0:
                sigma2[t] = max(initial_variance, 1e-12)

        return sigma2

    def _sequential_garch_forecast_percent(self, returns_percent: pd.Series) -> pd.Series:
        """Forecast each test date using information available before that date."""
        assert self.params_ is not None
        assert self.last_epsilon_ is not None
        assert self.last_sigma2_ is not None

        omega = self.params_.omega
        alpha = self.params_.alpha
        beta = self.params_.beta
        mu = self.params_.mu

        prev_eps = float(self.last_epsilon_)
        prev_sigma2 = float(self.last_sigma2_)
        forecasts = []

        for r_t in returns_percent.to_numpy(dtype=float):
            forecast_sigma2 = omega + alpha * prev_eps**2 + beta * prev_sigma2
            forecast_sigma2 = max(float(forecast_sigma2), 1e-12)
            forecasts.append(np.sqrt(forecast_sigma2))

            # Today's shock updates tomorrow's forecast, not today's forecast.
            prev_eps = float(r_t - mu)
            prev_sigma2 = forecast_sigma2

        return pd.Series(forecasts, index=returns_percent.index, name="garch_forecast_vol_percent")

    def _realized_volatility_proxy(
        self,
        df: pd.DataFrame,
        annualize_output: bool,
    ) -> np.ndarray:
        method = self.realized_vol_method.lower()

        if method in {"abs_return", "squared_return"}:
            returns_percent = self._compute_prediction_returns_percent(df)
            realized_daily_decimal = np.abs(returns_percent.to_numpy(dtype=float) / 100.0)
        elif method == "parkinson":
            if "High" not in df.columns or "Low" not in df.columns:
                raise ValueError("Parkinson volatility requires `High` and `Low` columns.")

            high = df["High"].astype(float).to_numpy(dtype=float)
            low = df["Low"].astype(float).to_numpy(dtype=float)
            if np.any(high <= 0) or np.any(low <= 0):
                raise ValueError("High and Low prices must be positive for Parkinson volatility.")

            realized_daily_decimal = np.sqrt((np.log(high / low) ** 2) / (4.0 * np.log(2.0)))
        else:
            raise ValueError(
                "realized_vol_method must be one of: `abs_return`, `squared_return`, `parkinson`."
            )

        if annualize_output:
            realized_daily_decimal = realized_daily_decimal * np.sqrt(self.trading_days)
        return realized_daily_decimal
