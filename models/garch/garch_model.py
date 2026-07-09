import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class fit():
    """
    Class to fit the GARCH(1,1) model to the provided data and make predictions
    for the specified date range.
    """

    def __init__(self, data, split_date):
        """
        Initialize the fit class with the provided data and prediction date range.
        data: DataFrame, historical stock data with multi-level columns
        split_date: str, first date for prediction in 'YYYY-MM-DD' format
        """

        self.data = data[data.index < split_date]
        self.ticker = self.data.columns.get_level_values(1)[0]
        self.start_date = data.index[0].strftime("%Y-%m-%d")
        self.start_date_predict = split_date
        self.end_date_predict = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _get_close_prices(self):
        """
        Extracts the close-price series from the multi-level dataframe.
        Returns: A pandas Series of close prices.
        """

        C = self.data[("Close", self.ticker)]
        C = C.dropna()
        C = C[C > 0]
        return C

    def _log_returns(self):
        """
        Computes daily log returns from close prices.
        Returns: A pandas Series of daily log returns in percent.
        """

        C = self._get_close_prices()
        returns = 100 * np.log(C / C.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def _garch_recursion(self, eps, omega, alpha, beta, initial_variance):
        """
        Computes the conditional variance sequence for GARCH(1,1).
        """

        sigma2 = np.zeros(len(eps))
        sigma2[0] = initial_variance

        for t in range(1, len(eps)):
            sigma2[t] = omega + alpha * eps[t - 1]**2 + beta * sigma2[t - 1]
            if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
                sigma2[t] = initial_variance

        return sigma2

    def volatility_fit(self):
        """
        Fits a GARCH(1,1) model to daily log returns.
        Returns: A dictionary containing fitted GARCH parameters.
        """

        returns = self._log_returns()

        if len(returns) < 30:
            raise ValueError("Not enough observations to fit the GARCH model.")

        mu = returns.mean()
        eps = returns - mu
        eps_values = eps.values

        initial_variance = np.var(eps_values)

        def negative_log_likelihood(params):
            omega, alpha, beta = params

            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
                return 1e10

            sigma2 = self._garch_recursion(
                eps_values,
                omega,
                alpha,
                beta,
                initial_variance
            )

            sigma2 = np.maximum(sigma2, 1e-10)

            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(sigma2) + eps_values**2 / sigma2
            )

            return -log_likelihood

        starting_values = np.array([
            0.05 * initial_variance,
            0.05,
            0.90
        ])

        bounds = [
            (1e-10, None),
            (1e-8, 0.999),
            (1e-8, 0.999)
        ]

        result = minimize(
            negative_log_likelihood,
            starting_values,
            method="L-BFGS-B",
            bounds=bounds
        )

        omega, alpha, beta = result.x

        fitted = {
            "mu": mu,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "last_epsilon": eps_values[-1],
            "last_sigma2": self._garch_recursion(
                eps_values,
                omega,
                alpha,
                beta,
                initial_variance
            )[-1],
            "success": result.success
        }

        return fitted

    def volatility_nextday(self):
        """
        Computes the volatility for the next day based on the fitted GARCH model.
        Returns: A float representing the annualized predicted volatility.
        """

        fitted = self.volatility_fit()

        omega = fitted["omega"]
        alpha = fitted["alpha"]
        beta = fitted["beta"]
        last_epsilon = fitted["last_epsilon"]
        last_sigma2 = fitted["last_sigma2"]

        next_sigma2 = omega + alpha * last_epsilon**2 + beta * last_sigma2
        next_sigma2 = max(next_sigma2, 1e-10)

        vol = np.sqrt(next_sigma2) / 100 * np.sqrt(252)

        return vol

    def trading_days_test(self):
        """
        Uses the CustomBusinessDay calendar to count the number of trading days
        between the start and end dates for prediction.
        Returns: An integer representing the number of trading days.
        """

        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

        trading_days = pd.date_range(
            start=self.start_date_predict,
            end=self.end_date_predict,
            freq=us_bd
        )

        return len(trading_days)

    def test(self):
        """
        Uses the fitted GARCH(1,1) model to predict volatility for each trading
        day in the specified prediction date range.
        Returns: A list of predicted annualized volatilities.
        """

        fitted = self.volatility_fit()

        omega = fitted["omega"]
        alpha = fitted["alpha"]
        beta = fitted["beta"]
        last_epsilon = fitted["last_epsilon"]
        last_sigma2 = fitted["last_sigma2"]

        future = []
        n = self.trading_days_test()

        prev_epsilon2 = last_epsilon**2
        prev_sigma2 = last_sigma2

        for _ in range(n):
            next_sigma2 = omega + alpha * prev_epsilon2 + beta * prev_sigma2
            next_sigma2 = max(next_sigma2, 1e-10)

            vol = np.sqrt(next_sigma2) / 100 * np.sqrt(252)
            future.append(vol)

            # For multi-step forecasts, E[epsilon^2] equals the predicted variance.
            prev_epsilon2 = next_sigma2
            prev_sigma2 = next_sigma2

        return future

    def volatility_avg(self):
        """
        Computes the average volatility over the prediction period.
        Returns: A float representing the average predicted volatility.
        """

        predictions = self.test()

        if len(predictions) == 0:
            return np.nan

        return np.mean(predictions)
