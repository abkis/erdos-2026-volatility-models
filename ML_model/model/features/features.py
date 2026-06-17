from typing import List

import numpy as np
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

class Features:
    """
        Calculates features ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
    """

    def __init__(self, df, window : int, lookback : int, lamb : float):
        """
            Input df with Close/Open/High/Low/Volume info
            Assume df grouped by Symbol/Date
            window: positive integer for rolling window
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
            Calculates features
        """
        self.df = df.copy()
        self.window = window
        self.lookback = lookback
        self.lamb = lamb

    def calculate_features(self, features : List):
        """
            Calculates features
            Returns only relevant ones
        """
        self._calculate_features()
        return self.df[features]

    def calculate_target(self, target : str):
        self.df[target] = (
            self.df.groupby('Symbol')['log_return']
              .transform(lambda x: x[::-1]
                           .rolling(self.window)
                           .std()[::-1]
                           .shift(-self.window + 1)
                           )
        )
        return self.df[target]
    
    @staticmethod
    def _ewma(r : float, lamb : float) -> float:
        """
            Helper method for calculating ewma
        """
        var = r.pow(2).ewm(alpha=1-lamb).mean()
        return np.sqrt(var * 252)

    @staticmethod
    def _ta_features(self, group):
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
                window=self.window,
            ).chaikin_money_flow()
        )
    
        return group

    def _calculate_features(self):
        """
            Calculate features used for RF later
            May only end up using a subset of these
        """
        self.df["log_return"] = (
            self.df.groupby("Symbol")["Close"]
              .transform(lambda x: np.log(x / x.shift(1)))
        )
        self.df["rolling_vol"] = (
            self.df.groupby("Symbol")["Close"]
              .transform(lambda x: x.rolling(self.window).std()*np.sqrt(252))
        )
        # high/low ratio used for parkinson volatility
        hl = np.log(self.df["High"] / self.df["Low"])**2

        self.df["parkinson_vol"] = (
            hl.groupby(self.df["Symbol"])
              .transform(
                  lambda x: np.sqrt(
                      x.rolling(self.window).mean() / (4*np.log(2))
                  ) * np.sqrt(252)
              )
        )
        # close-close
        self.df["close_close_vol"] = (
        self.df.groupby("Symbol")["log_return"]
          .transform(lambda x: x.rolling(self.window).std() * np.sqrt(252))
        )
        # garman-klass volatility
        gk_var = (
            0.5*np.log(self.df["High"]/self.df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.df["Close"]/self.df["Open"])**2
        )
        
        self.df["GK_vol"] = (
            gk_var.groupby(self.df["Symbol"])
                  .transform(
                      lambda x: np.sqrt(
                          x.rolling(self.window).mean()
                      ) * np.sqrt(252)
                  )
        )
        # GKYZ for overnight jumps
        prev_close = (
            self.df.groupby("Symbol")["Close"]
            .shift(1)
        )
        
        gkyz_var = (
            np.log(self.df["Open"]/prev_close)**2
            + 0.5*np.log(self.df["High"]/self.df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.df["Close"]/self.df["Open"])**2
        )
        
        self.df["GKYZ"] = (
            gkyz_var.groupby(self.df["Symbol"])
                    .transform(
                        lambda x: np.sqrt(
                            x.rolling(self.window).mean()
                        ) * np.sqrt(252)
                    )
        )

        # VIX proxy
        highest_close = (
            self.df.groupby("Symbol")["Close"]
            .transform(lambda x: x.rolling(self.lookback).max())
        )
        
        self.df["vix_fix"] = (
            100 *
            (highest_close - self.df["Low"])
            / highest_close
        )

        # ewma 
        self.df["ewma"] = (
        self.df.groupby("Symbol")["log_return"]
          .transform(lambda x : self._ewma(x, self.lamb))
        )

        # add indicators from ta library
        self.df = (
            self.df
            .groupby("Symbol", group_keys=False)
            .apply(self._ta_features)
        )
