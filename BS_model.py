import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class fit():
    '''
    Black-Scholes constant-volatility model.
    BS assumes volatility is constant, so this serves as the
    constant-vol baseline the other models are compared against.
    '''

    def __init__(self,data,start_date_predict,end_date_predict):
        '''
        Initialize with training data and prediction date range.

        data: DataFrame, historical stock data with multi-level columns
        start_date_predict: str, start date for prediction in 'YYYY-MM-DD' format
        end_date_predict: str, end date for prediction in 'YYYY-MM-DD' format
        '''

        self.data = data
        self.ticker = self.data.columns.get_level_values(1)[0]
        self.start_date_predict = start_date_predict
        self.end_date_predict = end_date_predict

    def volatility_fit(self):
        '''
        Estimates the constant annualized volatility from the
        training-period daily returns.

        Returns: float, the constant volatility used for every test day.
        '''

        C = self.data[("Close", self.ticker)]
        returns = C.pct_change().dropna()
        sigma = returns.std() * np.sqrt(252)
        return sigma

    def trading_days_test(self):
        '''
        Counts the number of US trading days between the prediction
        start and end dates using a Federal Holiday-aware calendar.

        Returns: int, number of trading days in the prediction period.
        '''

        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        trading_days = pd.date_range(
            start=self.start_date_predict,
            end=self.end_date_predict,
            freq=us_bd
        )
        return len(trading_days)

    def test(self):
        '''
        Returns the constant Black-Scholes volatility repeated for
        every trading day in the prediction period.

        Returns: list of predicted volatilities (same value repeated).
        '''

        sigma = self.volatility_fit()
        n = self.trading_days_test()
        return [sigma] * n