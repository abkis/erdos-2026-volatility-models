from socket import close

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


class dowloand_data:
    def __init__(self, ticker, start_date, end_date, testing_start, testing_end):
        '''
        Initialize the dowloand_data class with the provided parameters.
        ticker: str, stock ticker symbol
        start_date: str, start date for training data in 'YYYY-MM-DD' format
        end_date: str, end date for training data in 'YYYY-MM-DD' format
        testing_start: str, start date for testing data in 'YYYY-MM-DD' format
        testing_end: str, end date for testing data in 'YYYY-MM-DD' format
        '''

        self.ticker = ticker
        self.start_date=start_date
        self.end_date = end_date
        self.testing_start = testing_start
        self.testing_end = testing_end


    def training_data(self):
        '''
        Downloads the training data for the specified ticker and date range.
        Returns: A DataFrame containing the training data with multi-level columns.
        '''
        return yf.download(
            self.ticker, 
            start = self.start_date,
            end = self.end_date
        )

    def test_data(self):
        '''
        Downloads the testing data for the specified ticker and date range.
        Returns: A DataFrame containing the testing data with multi-level columns.
        '''

        return yf.download(
            self.ticker, 
            start = self.testing_start,
            end = self.testing_end
        )
    
    def rolling_volatility(self):
        '''
        Computes the Real rolling volatility for the specified ticker over the testing period.
        '''

        train_prices = self.training_data()[("Close", self.ticker)].tail(23)
        test_prices = self.test_data()[("Close", self.ticker)]

        # Need enough history before test starts
        prices = pd.concat([train_prices, test_prices])

        returns = prices.pct_change()
        rolling_vol = returns.rolling(22).std() * np.sqrt(252)

        # return only test dates
        return rolling_vol.loc[test_prices.index]



class compare_models:
    '''
    Class to compare different volatility models against the rolling volatility of a stock. 
    '''

    def __init__(self, ticker, models, rolling_vol, start_date_predict, end_date_predict):
        '''
        Initialize the compare_models class with the provided parameters.
        ticker: str, stock ticker symbol
        models: dict, a dictionary of volatility models
        rolling_vol: pd.DataFrame, the rolling volatility data
        start_date_predict: str, start date for prediction data in 'YYYY-MM-DD' format
        end_date_predict: str, end date for prediction data in 'YYYY-MM-DD' format
        '''

        self.ticker = ticker
        self.models = models
        self.rolling_vol = rolling_vol.dropna()
        self.start_date_predict = start_date_predict
        self.end_date_predict = end_date_predict
        

    #### Plot and compare models
    def plot_models(self):
        '''
        Plots the predicted volatility from different models against the rolling volatility of the stock.
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

    def MSE(self):
        '''
        Computes the Mean Squared Error (MSE) for each model against the rolling volatility.
        Returns: A dictionary containing the MSE for each model.
        '''

        mse = {}

        for name, model in self.models:
            N = min(len(model), len(self.rolling_vol))
            mse[name] = np.mean((model[:N] - self.rolling_vol[:N]) ** 2)

        return mse
    
    
    def QLIKE(self):
        '''
        Computes the QLIKE (Quadratic Loss) for each model against the rolling volatility.
        Returns: A dictionary containing the QLIKE for each model.
        '''

        qlike = {}

        for name, model in self.models:
            N = min(len(model), len(self.rolling_vol))
            qlike[name] = np.mean((self.rolling_vol[:N] / model[:N]) - np.log(self.rolling_vol[:N] / model[:N]) - 1)

        return qlike
    

    def plot_tables(self):

        '''
        Plots a table comparing the MSE and QLIKE metrics for each model and each ticker. This is only if you want a table for one of the ticker individually
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
            # Format every float inline to 5 decimal places right inside the call
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


def plot_table_all(all_results):
    results_df = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(results_df)))
    ax.axis("off")

    display_df = results_df.map(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x)

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=[f"{ticker} {metric}" for ticker, metric in results_df.columns],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Model Performance Comparison")
    plt.show()    

 