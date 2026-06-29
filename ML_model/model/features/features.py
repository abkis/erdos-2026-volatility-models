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
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf", "rv_1", "rv_5", "rv_21", "rv_63"]
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

    def calculate_target(self, target_name: str, target_horizon: int):

        if target_horizon <= 0:
            self.df["target_" + target_name] = self.df[target_name].shift(-1)
        else:
            self.df["target_" + target_name] = (
                self.df[target_name]
                .transform(
                    lambda x: np.sqrt(
                        252 *
                        x.pow(2)
                         .rolling(target_horizon)
                         .mean()
                         .shift(-target_horizon)
                    )
                )
            )
    
        return self.df["target_" + target_name]
    
    @staticmethod
    def _ewma(r : float, lamb : float) -> float:
        """
            Helper method for calculating ewma
        """
        var = r.pow(2).ewm(alpha=1-lamb).mean()
        return np.sqrt(var * 252)

    def _ta_features(self):
        """
            Calculate ta features
            Need Close/High/Low/Volume data
            Creates columns ["macd", "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
            Modifies self.df
        """

        # data 
        close = self.df["Close"]
        high = self.df["High"]
        low = self.df["Low"]
        volume = self.df["Volume"]

        # macd
        macd = MACD(
            close=close,
            window_fast=12,
            window_slow=26,
            window_sign=9,
        )
    
        self.df["macd"] = macd.macd()
        self.df["macd_signal"] = macd.macd_signal()
        self.df["macd_hist"] = macd.macd_diff()

        # rsi
        self.df["rsi"] = RSIIndicator(
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
    
        self.df["adx"] = adx.adx()

        # atr
        atr = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=14,
        )
    
        self.df["atr"] = atr.average_true_range()
    
        # bb_width
        bb = BollingerBands(
            close=close,
            window=20,
            window_dev=2,
        )

        self.df["bb_width"] = (
            bb.bollinger_hband()
            - bb.bollinger_lband()
        ) / bb.bollinger_mavg()

        # obv
        self.df["obv"] = (
            OnBalanceVolumeIndicator(
                close=close,
                volume=volume,
            ).on_balance_volume()
        )

        # cmf
        self.df["cmf"] = (
            ChaikinMoneyFlowIndicator(
                high=high,
                low=low,
                close=close,
                volume=volume,
                window=self.window,
            ).chaikin_money_flow()
        )
    

    def _calculate_features(self):
        """
            Calculate features used for RF later
            May only end up using a subset of these
        """
        self.df["log_return"] = (
            self.df["Close"]
              .transform(lambda x: np.log(x / x.shift(1)))
        )
        self.df["rolling_vol"] = (
            self.df["Close"]
              .transform(lambda x: x.rolling(self.window).std()*np.sqrt(252))
        )
        # high/low ratio used for parkinson volatility
        hl = np.log(self.df["High"] / self.df["Low"])**2

        self.df["parkinson_vol"] = (
            hl.transform(
                  lambda x: np.sqrt(
                      x.rolling(self.window).mean() / (4*np.log(2))
                  ) * np.sqrt(252)
              )
        )
        # close-close
        self.df["close_close_vol"] = (
        self.df["log_return"]
          .transform(lambda x: x.rolling(self.window).std() * np.sqrt(252))
        )
        # garman-klass volatility
        gk_var = (
            0.5*np.log(self.df["High"]/self.df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.df["Close"]/self.df["Open"])**2
        )
        
        self.df["GK_vol"] = (
            gk_var
                  .transform(
                      lambda x: np.sqrt(
                          x.rolling(self.window).mean()
                      ) * np.sqrt(252)
                  )
        )
        # GKYZ for overnight jumps
        prev_close = (
            self.df["Close"]
            .shift(1)
        )
        
        gkyz_var = (
            np.log(self.df["Open"]/prev_close)**2
            + 0.5*np.log(self.df["High"]/self.df["Low"])**2
            - (2*np.log(2)-1)
              * np.log(self.df["Close"]/self.df["Open"])**2
        )
        
        self.df["GKYZ"] = (
            gkyz_var
                    .transform(
                        lambda x: np.sqrt(
                            x.rolling(self.window).mean()
                        ) * np.sqrt(252)
                    )
        )

        # VIX proxy
        highest_close = (
            self.df["Close"]
            .transform(lambda x: x.rolling(self.lookback).max())
        )
        
        self.df["vix_fix"] = (
            100 *
            (highest_close - self.df["Low"])
            / highest_close
        )

        # ewma 
        self.df["ewma"] = (
        self.df["log_return"]
          .transform(lambda x : self._ewma(x, self.lamb))
        )

        # add indicators from ta library
        self._ta_features()

        # rolling volatility features
        for w in [1, 5, 21, 63]:
            self.df[f"rv_{w}"] = (
                self.df["log_return"]
                .transform(
                    lambda x: np.sqrt(
                        252 * x.pow(2).rolling(w).mean()
                    )
                )
            )
