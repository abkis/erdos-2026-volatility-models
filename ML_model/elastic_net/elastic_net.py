from matplotlib import pyplot as plt
import numpy as np
import yfinance as yf

from typing import List

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from model.refine_features.refine_features import RefineFeatures

class ElasticNetModel:
    """
        Handles logic of training/testing/evaluating/etc elastic net model for volatility predictions
    """

    def __init__(self, stocks : List[str], window : int, start_date : str, end_date : str, target_window : int, lookback = 22, lamb = 0.94, winsorize = True, corr_threshold = 0.9, lower_q=0.01, upper_q=0.99):
        """
            Initialize class
            stocks: list of stock names
            window: positive integer for rolling volatility window
            start_date, end_date: start/end date for stock info, format yyyy-mm-dd
            target_window: target is future realized volatility with window target_window. Should be an integer greater than zero
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
            winsorize: boolean that determines if outliers are clipped
            lower_q, upper_q are quantiles for winsorization
        """
        self.stocks = stocks
        self.window = window
        self.start_date = start_date
        self.end_date = end_date
        self.winsorize = winsorize
        self.corr_threshold = corr_threshold
        self.lookback = lookback
        self.lamb = lamb
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.target_window = target_window
        self.target = "target_real_vol_" + str(target_window)
        
        self.stock_data = yf.download(stocks, start=start_date, end=end_date)
        # stack stock data 
        self.stock_df = (
        self.stock_data.stack(level=1, future_stack=True)              
          .reset_index()
          .rename(columns={
              'level_0': 'Date',
              'Ticker': 'Symbol'
          })
        )
        # ensure properly sorted
        self.stock_df = self.stock_df.sort_values(["Symbol", "Date"])
        self.df = self.stock_df.copy()
        self.features = ["log_return", "rolling_vol", "rv_1", "rv_5", "rv_21", "rv_63", "parkinson_vol", "GK_vol", "abs_ret","range", "vol_change"]

        self.pred = None
        self.metrics = None
        self.model = None

    def _det_features(self):
        """
            Method for calculating features to be used/ensuring data is good
            Assumes cleaned data in self.stock_df
            Updates self.df
        """
        # calculate features
        self.df["log_return"] = (
            self.df.groupby("Symbol")["Close"]
              .transform(lambda x: np.log(x / x.shift(1)))
        )
        self.df["rolling_vol"] = (
            self.df.groupby("Symbol")["Close"]
              .transform(lambda x: x.rolling(self.window).std()*np.sqrt(252))
        )
        # lagged rv
        for w in [1, 5, 21, 63]:
            self.df[f"rv_{w}"] = (
                self.df.groupby("Symbol")["log_return"]
                .transform(
                    lambda x: np.sqrt(
                        252 * x.pow(2).rolling(w).mean()
                    )
                )
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
        self.df["abs_ret"] = (
            self.df.groupby("Symbol")["log_return"]
            .transform(lambda x: x.abs())
        )

        # volume info
        self.df["vol_change"] = (
            self.df.groupby("Symbol")["Volume"]
            .transform(lambda x: np.log(x / x.shift(1)))
        )

        self.df["range"] = (
            self.df.groupby("Symbol")
            .apply(lambda g: (g["High"] - g["Low"]) / g["Close"], include_groups=False)
            .reset_index(level=0, drop=True)
        )
        
    def _det_target(self):

        self.df[self.target] = (
            self.df.groupby("Symbol")["log_return"]
            .transform(
                lambda x: np.sqrt(
                    252 *
                    x.pow(2)
                     .rolling(self.target_window)
                     .mean()
                     .shift(-self.target_window)
                )
            )
        )
        
    def fit_model(self):
        """
            1. Cleans data
            2. Calculates features
            3. Fits model
        """
        self.df = self.df.sort_values(["Symbol", "Date"])
        self._det_features()
        self._det_target()
        self._clean_data()

        # train/test split
        cutoff_date = self.df["Date"].quantile(0.8)

        train = self.df[self.df["Date"] <= cutoff_date]
        test  = self.df[self.df["Date"] > cutoff_date]

        X_train = train[self.features]
        y_train = train[self.target]
        
        X_test = test[self.features]
        y_test = test[self.target]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.model = ElasticNetCV(cv=TimeSeriesSplit(5))
        self.model.fit(X_train, y_train)

        # model test results
        self.pred = self.model.predict(X_test)
        self.test = y_test

    def test_results(self):
        """
            After model is fit, see how it compares
            Calculate MAE, MSE, RMSE, MAPE, R2, rel_mse
        """
        mse = mean_squared_error(self.test, self.pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test, self.pred)
        mape = mean_absolute_percentage_error(self.test, self.pred)
        r2 = r2_score(self.test, self.pred)

        # relative data
        rel_mse = mse / np.var(self.test)

        self.metrics = {
            "mse" : mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape" : mape,
            "rel_mse" : rel_mse
        }
        return self.metrics
    
    def _clean_data(self):
        """
            Cleans data: 
                1. replace +/- infty with nan
                2. remove nan data
                3. clips outliers if winsorize set to true
        """

        # ensure properly sorted
        self.df = self.df.sort_values(["Symbol", "Date"])
        
        # replace infinities (if applicable)
        self.df = self.df.replace([np.inf, -np.inf], np.nan)

        # remove nan
        self.df = self.df.dropna(subset=self.features + [self.target])

        # winsorize (if applicable)
        if self.winsorize:
            for col in self.features:
                lower =self.df[col].quantile(self.lower_q)
                upper = self.df[col].quantile(self.upper_q)

                self.df[col] = self.df[col].clip(lower, upper)

    