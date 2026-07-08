
"""GARCH(1,1) volatility model.

This file supports two interfaces:

1. Project-level main notebook interface:

       fits = models.garch_model.fit(data, start_date_predict, end_date_predict)
       predicted_volatility = fits.test()

2. Direct model interface:

       model = GARCHModel(train_df, ticker="^GSPC")
       predicted_volatility = model.predict(test_df)
       metrics = model.test(test_df)

The input dataframe should contain at least a ``Close`` column. The shared
pipeline is expected to pass dataframes with ``Open``, ``Close``, ``High``,
``Low``, and ``Volume`` columns.
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

try:
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar,
        Holiday,
        GoodFriday,
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        USLaborDay,
        USThanksgivingDay,
        nearest_workday,
    )
    from pandas.tseries.offsets import CustomBusinessDay
except Exception:  # pragma: no cover
    AbstractHolidayCalendar = None
    CustomBusinessDay = None


if AbstractHolidayCalendar is not None:
    class _NYSEHolidayCalendar(AbstractHolidayCalendar):
        """Approximate NYSE holiday calendar for forecast horizon length."""

        rules = [
            Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
            Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]
else:  # pragma: no cover
    _NYSEHolidayCalendar = None


@dataclass
class GARCHParams:
    """Estimated GARCH(1,1) parameters."""

    mu: float
    omega: float
    alpha: float
    beta: float

    @property
    def alpha_plus_beta(self) -> float:
        return self.alpha + self.beta


class GARCHModel:
    """One-step-ahead GARCH(1,1) volatility forecaster."""

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
        """Estimate parameters from the training data."""
        if train_df is not None:
            self.train_df = self._prepare_dataframe(train_df)

        returns = self._compute_log_returns_percent(self.train_df)
        if len(returns) < self.min_obs:
            raise ValueError(
                f"Not enough observations to fit GARCH: need {self.min_obs}, got {len(returns)}."
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
        """Forecast volatility for rows in ``df`` using sequential updates."""
        self._check_fitted()

        pred_df = self._prepare_dataframe(df)
        pred_df = self._filter_dates(pred_df, start_date, end_date)
        if pred_df.empty:
            return np.array([], dtype=float)

        returns = self._compute_prediction_returns_percent(pred_df)
        pred_vol_percent = self._sequential_garch_forecast_percent(returns)
        pred_vol_decimal = pred_vol_percent / 100.0

        use_annualized = self.annualize_output if annualize_output is None else annualize_output
        if use_annualized:
            pred_vol_decimal = pred_vol_decimal * np.sqrt(self.trading_days)

        return pred_vol_decimal.to_numpy(dtype=float)

    def forecast_horizon(
        self,
        horizon: int,
        annualize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """Return multi-step-ahead volatility forecasts without future prices.

        This is the method used by the shared ``main.ipynb`` adapter below,
        because that notebook passes only training data into each model.
        """
        self._check_fitted()
        if horizon <= 0:
            return np.array([], dtype=float)

        assert self.params_ is not None
        assert self.last_epsilon_ is not None
        assert self.last_sigma2_ is not None

        omega = self.params_.omega
        alpha = self.params_.alpha
        beta = self.params_.beta

        prev_sigma2 = float(self.last_sigma2_)
        prev_eps2 = float(self.last_epsilon_ ** 2)

        forecasts = []
        for _ in range(int(horizon)):
            forecast_sigma2 = omega + alpha * prev_eps2 + beta * prev_sigma2
            forecast_sigma2 = max(float(forecast_sigma2), 1e-12)
            forecasts.append(np.sqrt(forecast_sigma2) / 100.0)

            # For horizons beyond one day, E[epsilon^2] equals forecast variance.
            prev_eps2 = forecast_sigma2
            prev_sigma2 = forecast_sigma2

        out = np.asarray(forecasts, dtype=float)
        use_annualized = self.annualize_output if annualize_output is None else annualize_output
        if use_annualized:
            out = out * np.sqrt(self.trading_days)
        return out

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
        """Return estimated parameters and convergence status."""
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
            if method == "squared_return":
                realized_daily_decimal = realized_daily_decimal**2
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


class fit:
    """Adapter for the shared ``main.ipynb`` interface.

    The group notebook calls each model as:

        fits = models.garch_model.fit(data, start_date_predict, end_date_predict)
        predicted_volatility = fits.test()

    This adapter keeps that interface while reusing ``GARCHModel`` internally.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        start_date_predict: str,
        end_date_predict: str,
        ticker: Optional[str] = None,
        annualize_output: bool = True,
        price_col: str = "Close",
        trading_days: int = 252,
    ) -> None:
        self.data = data
        self.start_date_predict = pd.to_datetime(start_date_predict)
        self.end_date_predict = pd.to_datetime(end_date_predict)
        self.ticker = ticker
        self.annualize_output = annualize_output
        self.price_col = price_col
        self.trading_days = trading_days

        self.model = GARCHModel(
            train_df=data,
            ticker=ticker,
            price_col=price_col,
            annualize_output=annualize_output,
            trading_days=trading_days,
        )

        self.prediction_dates = self._prediction_dates(
            self.start_date_predict,
            self.end_date_predict,
        )

    def test(self):
        """Return forecasts in the format expected by ``compare_models``."""
        forecasts = self.model.forecast_horizon(
            horizon=len(self.prediction_dates),
            annualize_output=self.annualize_output,
        )
        return [np.float64(x) for x in forecasts]

    def predict(self):
        """Alias for ``test``."""
        return self.test()

    def get_fitted_params(self) -> Dict[str, float]:
        return self.model.get_fitted_params()

    @staticmethod
    def _prediction_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        if end_date < start_date:
            return pd.DatetimeIndex([])

        if CustomBusinessDay is not None and _NYSEHolidayCalendar is not None:
            nyse_day = CustomBusinessDay(calendar=_NYSEHolidayCalendar())
            return pd.date_range(start=start_date, end=end_date, freq=nyse_day)

        return pd.bdate_range(start=start_date, end=end_date)
