from socket import close
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import os

#importing the models
import models.HAR_GK
import models.garch_model
import models.pdv_model
import models.RF_model
import models.BS_model
import importlib

## We can remove this in the final version, it helps me work on the main file and this simultaneously
importlib.reload(models.HAR_GK)
importlib.reload(models.garch_model)
importlib.reload(models.pdv_model)
importlib.reload(models.RF_model)
importlib.reload(models.BS_model)


'''
The file contains three main classes: download_data, single_ticker_compare, and compare_all_tickers.
- The download_data class is responsible for downloading historical stock data and computing rolling volatility.    
- The single_ticker_compare class is responsible for comparing different volatility models against the rolling volatility of a stock for a single ticker.
- The compare_all_tickers class is responsible for combining the info for all the ticker and plotting the graph and table for all the tickers and models together.
'''

class download_data:
    '''
    Class to download historical stock data and compute rolling volatility for a given ticker and date range.
    '''

    def __init__(self, ticker, date_range):
        '''
        Initialize the download_data class with the provided parameters.
        ticker: str, stock ticker symbol
        date_range: list, a list containing the start date, split date, and end date for testing data in 'YYYY-MM-DD' format
        '''

        self.ticker = ticker
        self.start_date = date_range[0]
        self.split_date = date_range[1]
        self.testing_end = date_range[2]


    def all_data(self):
        '''
        Downloads the training data for the specified ticker and date range.
        Returns: A DataFrame containing the training data with multi-level columns.
        '''
        return yf.download(
            self.ticker, 
            start = self.start_date,
            end = self.testing_end
        )


    
    def rolling_volatility(self):
        '''
        Computes the Real rolling volatility for the specified ticker over the testing period.
        '''
        # Need enough history before test starts
        prices = self.all_data()[("Close", self.ticker)]

        returns = prices.pct_change()
        rolling_vol = returns.rolling(22).std() * np.sqrt(252)

        # return only test dates
        return rolling_vol.loc[rolling_vol.index >= self.split_date]


class single_ticker_compare:
    '''
    Class to compare different volatility models against the rolling volatility of a stock. 
    '''

    def __init__(self, ticker, model_vol, rolling_vol, start_date_predict, end_date_predict, save_results=False):
        '''
        Initialize the compare_models class with the provided parameters.
        ticker: str, stock ticker symbol
        model_vol: dict, a dictionary of predicted volatility values
        rolling_vol: pd.DataFrame, the rolling volatility data
        start_date_predict: str, start date for prediction data in 'YYYY-MM-DD' format
        end_date_predict: str, end date for prediction data in 'YYYY-MM-DD' format
        '''

        self.ticker = ticker
        self.model_vol = model_vol
        self.rolling_vol = rolling_vol.dropna()
        self.start_date_predict = start_date_predict
        self.end_date_predict = end_date_predict
        self.save_results = save_results

    #### Plot and compare models
    def plot_models(self):
        '''
        Plots the predicted volatility from different models against the rolling volatility of the stock.
        '''

        plt.figure(figsize=(12,7))

        trading_days = self.rolling_vol.index

        for model_name in self.model_vol:
            N = min(len(self.model_vol[model_name]), len(self.rolling_vol))
            plt.plot(trading_days[:N], self.model_vol[model_name][:N], label=model_name)
           

        plt.plot(trading_days, self.rolling_vol, label="Rolling Volatility", color="black", linewidth=3)
        plt.grid(True)
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.title(f"Volatility Comparison for {self.ticker}")
        plt.legend()

        if self.save_results:
            folder = "results"
            os.makedirs(folder, exist_ok=True)

            plt.savefig(os.path.join(folder, f"{self.ticker}_volatility_comparison.png"),
                        dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()

    def MSE(self):
        '''
        Computes the Mean Squared Error (MSE) for each model against the rolling volatility.
        Returns: A dictionary containing the MSE for each model.
        '''

        mse = {}

        for model_name in self.model_vol:
            N = min(len(self.model_vol[model_name]), len(self.rolling_vol))
            mse[model_name] = np.mean((self.model_vol[model_name][:N] - self.rolling_vol[:N]) ** 2)

        return mse
    
    
    def QLIKE(self):
        '''
        Computes the QLIKE (Quadratic Loss) for each model against the rolling volatility.
        Returns: A dictionary containing the QLIKE for each model.
        '''

        qlike = {}

        for model_name in self.model_vol:
            N = min(len(self.model_vol[model_name]), len(self.rolling_vol))
            qlike[model_name] = np.mean((self.rolling_vol[:N] / self.model_vol[model_name][:N]) - np.log(self.rolling_vol[:N] / self.model_vol[model_name][:N]) - 1)

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
            # Format every float inline to 6 decimal places right inside the call
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
  

class compare_all_models:
    
    '''
    Class to compare different volatility models against the rolling volatility of a stock for multiple tickers. 
    '''

    def __init__(self, tickers, date_range, save_results=False):
        '''
        Initialize the compare_all_tickers class with the provided parameters.
        all_results: pd.DataFrame, a DataFrame containing the MSE and QLIKE metrics for each model and each ticker.
        '''
        self.tickers = tickers
        self.date_range = date_range
        self.save_results = save_results
        self.model_dic = {"HAR-GK": models.HAR_GK, 
                          "Black-Scholes": models.BS_model,
                "Path Dependent": models.pdv_model,
                "GARCH": models.garch_model,
                "ML-Models": models.RF_model
        }
        self.split_date = date_range[1]

    
        
    def compute_metrics(self):
        '''

        '''

        all_results = {}
        ## Running the loop for each ticker
        for ticker in self.tickers:

            ## Calling the download_data class to get the data for the ticker and date range.

            df = download_data(ticker, self.date_range)
            data = df.all_data()

            ## Running the for each model
            model_vol  = {}

            for model_name in self.model_dic:
                fits = self.model_dic[model_name].fit(data, self.split_date)
                predicted_volatility = fits.test()

                model_vol[model_name] = predicted_volatility
                # print(f"Model: {model_name}, Ticker: {ticker}, Predicted Volatility: {predicted_volatility}")
                
            Compare = single_ticker_compare(ticker, model_vol, df.rolling_volatility(), self.split_date, self.date_range[2],self.save_results)

            Compare.plot_models()
            all_results[(ticker, "MSE")] = Compare.MSE()
            all_results[(ticker, "QLIKE")] = Compare.QLIKE()

        self.plot_table_all(all_results)

    def plot_table_all(self, all_results):
        
        '''
        Plotting a table comparing the MSE and QLIKE metrics for each model and each ticker. This is only if you want a table for all tickers and models together
        all_results: pd.DataFrame, a DataFrame containing the MSE and QLIKE metrics
        '''

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

        plt.title("Model Performance Comparison MSE and QLIKE")
        plt.show()

        if self.save_results:
            filename = "model_performance_comparison.csv"
            folder = "results"
            os.makedirs(folder, exist_ok=True)

            results_df.to_csv(os.path.join(folder, filename), index=True)
            print(f"Results saved to {os.path.join(folder, filename)}")

    

            
