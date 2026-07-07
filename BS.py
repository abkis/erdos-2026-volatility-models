import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class fit():
    '''
    Class to fit the Black-Scholes constant-volatility model to the provided data and make predictions for the specified date range.
    BS assumes volatility is constant, so this serves as the constant-vol baseline the other models are compared against.
    '''

    def __init__(self,data,start_date_predict,end_date_predict):
        '''
        Initialize the fit class with the provided data and prediction date range.
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
        Estimates the constant annualized volatility from the training-period daily returns.
        Returns: A float representing the constant volatility used for every test day.
        '''

        C = self.data[("Close", self.ticker)]
        returns = C.pct_change().dropna()
        sigma = returns.std() * np.sqrt(252)
        return sigma

    def trading_days_test(self):
        '''
        Uses the CustomBusinessDay calendar to count the number of trading days between the start and end dates for prediction.
        Returns: An integer representing the number of trading days in the prediction period.
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
        Returns the constant Black-Scholes volatility for every trading day in the prediction period.
        Returns: A list of predicted volatilities for each trading day in the prediction period.
        '''

        sigma = self.volatility_fit()
        n = self.trading_days_test()
        return [sigma] * n

    def volatility_avg(self):
        '''
        Computes the average volatility over the prediction period.
        Returns: A float representing the average predicted volatility.
        '''

        return np.mean(self.test())


class compare_BS():
    '''
    Class to compare the BS model against the realized rolling volatility of a stock.
    Same interface as compare_models in main.py — same method names, same return types.
    '''

    def __init__(self,ticker,models,rolling_vol,start_date_predict,end_date_predict):
        '''
        Initialize the compare_BS class with the provided parameters.
        ticker: str, stock ticker symbol
        models: list of [name, predicted_vols] pairs, e.g. [["BS", [0.23, 0.23, ...]]]
        rolling_vol: pd.Series, realized rolling volatility over the test period
        start_date_predict: str, start date for prediction in 'YYYY-MM-DD' format
        end_date_predict: str, end date for prediction in 'YYYY-MM-DD' format
        '''

        self.ticker = ticker
        self.models = models
        self.rolling_vol = rolling_vol.dropna()
        self.start_date_predict = start_date_predict
        self.end_date_predict = end_date_predict

    def MSE(self):
        '''
        Computes the Mean Squared Error (MSE) for each model against the rolling volatility.
        Returns: A dictionary containing the MSE for each model.
        '''

        mse = {}

        for name, model in self.models:
            N = min(len(model), len(self.rolling_vol))
            mse[name] = np.mean((np.array(model[:N]) - self.rolling_vol.values[:N]) ** 2)

        return mse

    def QLIKE(self):
        '''
        Computes the QLIKE loss for each model against the rolling volatility.
        Returns: A dictionary containing the QLIKE for each model.
        '''

        qlike = {}

        for name, model in self.models:
            N = min(len(model), len(self.rolling_vol))
            qlike[name] = np.mean(
                (self.rolling_vol.values[:N] / np.array(model[:N]))
                - np.log(self.rolling_vol.values[:N] / np.array(model[:N]))
                - 1
            )

        return qlike

    def plot_models(self):
        '''
        Plots the predicted volatility from each model against the realized rolling volatility.
        '''

        plt.figure(figsize=(12,7))

        trading_days = self.rolling_vol.index

        for name, model in self.models:
            N = min(len(model), len(self.rolling_vol))
            plt.plot(trading_days[:N], model[:N], label=name)

        plt.plot(trading_days, self.rolling_vol, label="Rolling Volatility", color="black", linewidth=2)
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.title(f"Volatility Comparison for {self.ticker}")
        plt.legend()
        plt.show()

    def plot_tables(self):
        '''
        Plots a table comparing the MSE and QLIKE metrics for each model.
        '''

        mse = self.MSE()
        qlike = self.QLIKE()

        df = pd.DataFrame({
            "Model": list(mse.keys()),
            "MSE": list(mse.values()),
            "QLIKE": [qlike[name] for name in mse]
        })

        fig, ax = plt.subplots(figsize=(5, 2 + 0.5 * len(df)))
        ax.axis("off")

        table = ax.table(
            cellText=[[f"{x:.6f}" if isinstance(x, (int, float)) else x for x in row] for row in df.values],
            colLabels=df.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 1.5)

        plt.title(f"Model Performance Comparison for {self.ticker}")
        plt.show()
