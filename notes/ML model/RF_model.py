import yfinance as yf
import numpy as np
import pandas as pd

from typing import List

from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volume import (
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
)

class RF_Model:
    """
        Handles logic of training/testing/evaluating/etc RF model
        Handles data download and preprocessing
        Deals with daily historical data
    """

    def __init__(self, stocks : List[str], window : int, start_date : str, end_date : str, features : List[str]| None):
        """
            Initialize class
            stocks: list of stock names
            window: positive integer for rolling window
            start_date, end_date: start/end date for stock info, format yyyy-mm-dd
            features: list of strings, must be from ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
                If none defaults to full list
        """
        self.stocks = stocks
        self.window = window
        self.start_date = start_date
        self.end_date = end_date
        self.all_features = ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
        if features is not None:
            for feature in features:
                assert feature in self.all_features
            self.features = features
        else:
            self.features = self.all_features
        
        self.stock_data = yf.download(stock_names, start=start_date, end=end_date)
        # stack stock data 
        self.stock_df = (
        stock_data.stack(level=1)              
          .reset_index()
          .rename(columns={
              'level_0': 'Date',
              'Ticker': 'Symbol'
          })
        )

    @static_method
    def _ewma(r : float, lamb : float) -> float:
        """
            Helper method for calculating ewma
        """
        var = r.pow(2).ewm(alpha=1-lamb).mean()
        return np.sqrt(var * 252)

    @static_method
    def _ta_features(group):
        """
            Calculate ta features when data grouped by symbol
            Need Close/High/Low/Volume data
            Creates columns ["macd", "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
        """

        # data 
        close = group["Close"]
        high = group["High"]
        low = group["Low"]
        volume = group["Volume"]

        # macd
        macd = MACD(
            close=close,
            window_fast=12,
            window_slow=26,
            window_sign=9,
        )
    
        group["macd"] = macd.macd()
        group["macd_signal"] = macd.macd_signal()
        group["macd_hist"] = macd.macd_diff()

        # rsi
        group["rsi"] = RSIIndicator(
            close=close,
            window=14,
        ).rsi()
    
        # adx
        adx = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=14,
        )
    
        group["adx"] = adx.adx()

        # atr
        atr = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=14,
        )
    
        group["atr"] = atr.average_true_range()
    
        # bb_width
        bb = BollingerBands(
            close=close,
            window=20,
            window_dev=2,
        )

        group["bb_width"] = (
            bb.bollinger_hband()
            - bb.bollinger_lband()
        ) / bb.bollinger_mavg()

        # obv
        group["obv"] = (
            OnBalanceVolumeIndicator(
                close=close,
                volume=volume,
            ).on_balance_volume()
        )

        # cmf
        group["cmf"] = (
            ChaikinMoneyFlowIndicator(
                high=high,
                low=low,
                close=close,
                volume=volume,
                window=20,
            ).chaikin_money_flow()
        )
    
        return group

    def calculate_features(self, lookback=22, lamb = 0.94):
        """
            Calculate features used for RF later
            May only end up using a subset of these
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
        """
        self.stock_df["log_return"] = (
            self.stock_df.groupby("Symbol")["Close"]
              .transform(lambda x: np.log(x / x.shift(1)))
        )
        self.stock_df["rolling_vol"] = (
            self.stock_df.groupby("Symbol")["Close"]
              .transform(lambda x: x.rolling(self.window).std()*np.sqrt(252))
        )
        # high/low ratio used for parkinson volatility
        hl = np.log(self.stock_df["High"] / self.stock_df["Low"])**2

        self.stock_df["parkinson_vol"] = (
            hl.groupby(self.stock_df["Symbol"])
              .transform(
                  lambda x: np.sqrt(
                      x.rolling(self.window).mean() / (4*np.log(2))
                  ) * np.sqrt(252)
              )
        )
        # close-close
        self.stock_df["close_close_vol"] = (
        self.stock_df.groupby("Symbol")["log_ret"]
          .transform(lambda x: x.rolling(self.window).std() * np.sqrt(252))
        )
        # garman-klass volatility
        gk_var = (
            0.5*np.log(self.stock_df["High"]/self.stock_df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.stock_df["Close"]/self.stock_df["Open"])**2
        )
        
        self.stock_df["GK_vol"] = (
            gk_var.groupby(self.stock_df["Symbol"])
                  .transform(
                      lambda x: np.sqrt(
                          x.rolling(self.window).mean()
                      ) * np.sqrt(252)
                  )
        )
        # GKYZ for overnight jumps
        prev_close = (
            self.stock_df.groupby("Symbol")["Close"]
            .shift(1)
        )
        
        gkyz_var = (
            np.log(self.stock_df["Open"]/prev_close)**2
            + 0.5*np.log(self.stock_df["High"]/self.stock_df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.stock_df["Close"]/self.stock_df["Open"])**2
        )
        
        self.stock_df["GKYZ"] = (
            gkyz_var.groupby(self.stock_df["Symbol"])
                    .transform(
                        lambda x: np.sqrt(
                            x.rolling(self.window).mean()
                        ) * np.sqrt(252)
                    )
        )

        # VIX proxy
        highest_close = (
            self.stock_df.groupby("Symbol")["Close"]
            .transform(lambda x: x.rolling(lookback).max())
        )
        
        self.stock_df["vix_fix"] = (
            100 *
            (highest_close - self.stock_df["Low"])
            / highest_close
        )

        # ewma 
        self.stock_df["ewma"] = (
        self.stock_df.groupby("Symbol")["log_ret"]
          .transform(lambda x : self._ewma(x, lamb))
        )

        # add indicators from ta library
        self.stock_df = (
            self.stock_df
            .groupby("Symbol", group_keys=False)
            .apply(self._ta_features)
        )