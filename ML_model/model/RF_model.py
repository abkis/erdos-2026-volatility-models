from matplotlib import pyplot as plt
import numpy as np
import yfinance as yf

from typing import List

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model.features.features import Features
from model.refine_features.refine_features import RefineFeatures

class RF_Model:
    """
        Handles logic of training/testing/evaluating/etc RF model
        Handles data download and preprocessing
        Deals with daily historical data
    """

    def __init__(self, stocks : List[str], window : int, start_date : str, end_date : str, features : List[str]| None, target_window : int,
                lookback = 22, lamb = 0.94, winsorize = True, corr_threshold = 0.9, lower_q=0.01, upper_q=0.99, rf_refine=True, grid_search = False, rf_params = None):
        """
            Initialize class
            stocks: list of stock names
            window: positive integer for rolling window
            start_date, end_date: start/end date for stock info, format yyyy-mm-dd
            features: list of strings, must be from ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf", "rv_1", "rv_5", "rv_21", "rv_63"]
                If none defaults to full list
            target_window: target is future realized volatility with window target_window. Should be an integer greater than one
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
            winsorize: boolean that determines if outliers are clipped
            corr_threshold: float value used to determine which features are dropped in case of high correlation
            lower_q, upper_q are quantiles for winsorization
            performs additional rf refinement if rf_refine is True
            grid_search: if set to true runs grid search to get best params
            rf_params: used as paramaters for rf. if none given use default
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
        self.rf_refine = rf_refine
        self.all_features = ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf", "rv_1",  "rv_5", "rv_21", "rv_63"]
        if features is not None:
            for feature in features:
                assert feature in self.all_features
            self.features = features
        else:
            self.features = self.all_features

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

        self.final_features = self.features 
        self.final_df = self.stock_df.copy()

        self.rf = None
        self.pred = None
        self.metrics = None

        if rf_params is None:
            self.rf_params = {
                "n_estimators": 500,
                "max_depth": 8,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1
            }
        else:
            self.rf_params = rf_params
        
        self.grid_search = grid_search

    def det_features(self):
        """
            Method for calculating features to be used/ensuring data is good
            1. Uses Features class to calculate features
            2. Cleans data
            3. Uses Refine_Features class to refine features based on correlation info
            4. Updates self.final_features and self.final_df
        """
        # calculate features
        features_class = Features(self.final_df, self.window, self.lookback, self.lamb)
        self.final_df[self.features] = features_class.calculate_features(self.features)
        self.final_df[self.target] = features_class.calculate_target(self.target_window, self.target)

        # clean data
        self._clean_data()

        # refine features based on correlation info
        refine_class = RefineFeatures(self.final_df, self.features, self.target, self.corr_threshold, self.rf_refine)
        self.final_features, _ = refine_class.refine_features()

    def fit_RF(self):
        """
            Assumes self.final_features and self.final_df correctly updated
            Fits random forest
        """
        data = self.final_df.dropna(subset=self.final_features + [self.target])

        X = data[self.final_features]        
        y = data[self.target]
        
        meta = data[['Date', 'Symbol']] # keep track of metadata
        
        # train/test split
        split_date = data.Date.quantile(0.8)
        
        train_mask = data['Date'] < split_date
        
        X_train = X[train_mask]
        X_test  = X[~train_mask]
        
        y_train = y[train_mask]
        y_test  = y[~train_mask]
        
        meta_test = meta[~train_mask]

        if self.grid_search:
            # get best params for rf

            # setup model
            rf = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        
            # pick best model using gridsearch
            
            param_grid = {
                "n_estimators": [300, 500],
                "max_depth": [5, 8, 12, None],
                "min_samples_leaf": [1, 5, 10, 20],
                "max_features": ["sqrt", 0.5]
            }
        
            tscv = TimeSeriesSplit(n_splits=5)
        
            grid = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1
            )
        
            grid.fit(X_train, y_train)
        
            # best model
            self.rf = grid.best_estimator_
            self.rf_params = grid.best_params_
        else:
            self.rf = RandomForestRegressor(**self.rf_params)
            self.rf = self.rf.fit(X_train, y_train)
        
        # test model
        self.pred = self.rf.predict(X_test)
        self.test = y_test
    
        self.results = meta_test.copy()
        self.results["predicted_vol"] = self.pred
        self.results["realized_vol"] = y_test.values

    def test_results(self):
        """
            After RF is fit, see how it compares
            Calculate MAE, MSE, RMSE, MAPE, R2, and bias
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

    def plot_results(self):
        """
            plot actual vs predicted results
        """
        results = self.results.sort_values('Date')

        fig, axes = plt.subplots(len(self.stocks), 1, figsize=(12, 4 * len(self.stocks)), sharex=True)
        
        if len(self.stocks) == 1:
            axes = [axes]
        
        for i, name in enumerate(self.stocks):
            ax = axes[i]
        
            stock_data = results[results["Symbol"] == name].dropna(subset=["realized_vol", "predicted_vol"])
        
            ax.plot(stock_data["Date"], stock_data["realized_vol"], label="Realized Vol")
            ax.plot(stock_data["Date"], stock_data["predicted_vol"], label="Predicted Vol")
        
            ax.set_title(name)
            ax.set_ylabel("Volatility")
            ax.legend()
        
        axes[-1].set_xlabel("Date")
        
        plt.tight_layout()
        plt.show()

    def scatter(self):
        results = self.results.sort_values("Date")
        plt.figure(figsize=(6,6))

        plt.scatter(
            results["realized_vol"],
            results["predicted_vol"],
            alpha=0.25
        )
        
        minv = min(results["realized_vol"].min(), results["predicted_vol"].min())
        maxv = max(results["realized_vol"].max(), results["predicted_vol"].max())
        
        plt.plot([minv, maxv], [minv, maxv], "r--")
        
        plt.title("Predicted vs Realized Volatility")
        plt.xlabel("Realized Volatility")
        plt.ylabel("Predicted Volatility")
        plt.show()

    def rolling_mae_plot(self):
        results = self.results.sort_values("Date")
        results["abs_error"] = (
            results["predicted_vol"] - results["realized_vol"]
        ).abs()
        
        results["rolling_mae"] = (
            results.groupby("Symbol")["abs_error"]
            .transform(lambda x: x.rolling(50).mean())
        )
        
        plt.figure(figsize=(12,5))
        
        for name in self.stocks:
            stock = results[results["Symbol"] == name]
            plt.plot(stock["Date"], stock["rolling_mae"], label=name)
        
        plt.title("Rolling MAE")
        plt.xlabel("Date")
        plt.ylabel("MAE")
        plt.legend()
        plt.show()

    def hist(self):
        results = self.results.sort_values("Date")
        plt.figure(figsize=(8,5))

        plt.hist(results["predicted_vol"] - results["realized_vol"], bins=50, alpha=0.7)
        
        plt.title("Prediction Error Distribution (Pred - Real)")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.show()
    
    def _clean_data(self):
        """
            Cleans data: 
                1. replace +/- infty with nan
                2. remove nan data
                3. clips outliers if winsorize set to true
        """

        # ensure properly sorted
        self.final_df = self.final_df.sort_values(["Symbol", "Date"])
        
        # replace infinities (if applicable)
        self.final_df = self.final_df.replace([np.inf, -np.inf], np.nan)

        # remove nan
        self.final_df = self.final_df.dropna(subset=self.features + [self.target])

        # winsorize (if applicable)
        if self.winsorize:
            for col in self.features:
                lower =self.final_df[col].quantile(self.lower_q)
                upper = self.final_df[col].quantile(self.upper_q)

                self.final_df[col] = self.final_df[col].clip(lower, upper)

    