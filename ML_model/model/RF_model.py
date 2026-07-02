import numpy as np
import pandas as pd

from typing import List

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from model.features.features import Features
from model.refine_features.refine_features import RefineFeatures

class RF_Model:
    """
        Handles logic of training/testing/evaluating/etc RF model
        Assumed input is cleaned dataframe with stock data which corresponds to training data
            Required column names: Date, Open, Close, Volume, High, Low
        Deals with daily historical data
    """

    def __init__(self, features : List[str]| None, window: int=21, target_name : str = "GK_vol", target_window = 0,  lookback = 22, lamb = 0.94, 
                 winsorize = False, corr_threshold = 0.9, lower_q=0.01, upper_q=0.99, rf_refine=True, grid_search = False, rf_params = None):
        """
            Initialize class
            window: positive integer for rolling window
            features: list of strings, must be from ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf", "rv_1", "rv_5", "rv_21", "rv_63"]
                If none defaults to full list
            target_col: name of feature which will be target. Must be one of the features above
            target_window: if set to zero assume looking at next-day predictions. Else target is calculated as a window
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
            winsorize: clips data/deals with outliers if set to True
            corr_threshold: float value used to determine which features are dropped in case of high correlation
            lower_q, upper_q are quantiles for winsorization
            performs additional rf refinement if rf_refine is True
            grid_search: if set to true runs grid search to get best params
            rf_params: used as paramaters for rf. if none given use default
        """
        self.df = None

        self.start_test = None
        self.end_test = None
        self.window = window
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
        assert(target_name in self.features)
        self.target_name = target_name

        self.final_features = self.features 

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

    def fit(self, df : pd.DataFrame, start_test : str, end_test =None):
        """
            Fit model
            df: pandas dataframe with column names Date, Open, Close, Volume, High, Low
                - Assumed to be appropriately cleaned, outliers dealt with, etc
                - Used for training of model
            start_test denotes start of test dates
            end_test denotes end of test dates
        """
        self.df = df.sort_values(by=["Date"]) 
        self.start_test = start_test
        self.end_test = end_test
        self.det_features()
        self.fit_RF()
        return self

    def det_features(self):
        """
            Method for calculating features to be used/ensuring data is good
            1. Uses Features class to calculate features
            2. Cleans data
            3. Uses Refine_Features class to refine features based on correlation info
            4. Updates self.final_features and self.final_df
        """
        # calculate features
        features_class = Features(self.df, self.window, self.lookback, self.lamb)
        self.df[self.features] = features_class.calculate_features(self.features)
        self.df["target_" + self.target_name] = features_class.calculate_target(self.target_name, self.target_window)

        # clean data
        self.df = self._clean_data(self.df)

        # refine features based on correlation info
        refine_class = RefineFeatures(self.df, self.features, "target_" + self.target_name, self.corr_threshold, self.rf_refine)
        self.final_features, _ = refine_class.refine_features()

    def fit_RF(self):
        """
            Assumes self.final_features and self.final_df correctly updated
            Fits random forest
            Uses all data as train data
        """
        self.df = self.df.sort_values(by=["Date"]) # ensure properly sorted
        data = self.df.dropna(subset=self.final_features + ["target_" + self.target_name])

        X = data[self.final_features]        
        y = data["target_" + self.target_name]

        train_mask = data['Date'] < self.start_test

        X_train = X[train_mask]
        X_test  = X[~train_mask]
        
        y_train = y[train_mask]
        y_test  = y[~train_mask]

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

        return self
        
    def test(self):
        return self.pred
        
    def test_results(self, test_data : pd.DataFrame):
        """
            After RF is fit, see how it compares with test data
            X_test: pandas dataframe with columns Date, Open, Close, High, Low, Volume, sorted by date
            Calculate MAE, MSE, RMSE, MAPE, R2, and relative mse
        """
        # ensure test data in correct format
        test_data = test_data.sort_values(by=["Date"])
        features_class = Features(test_data, self.window, self.lookback, self.lamb)
        test_data[self.final_features] = features_class.calculate_features(self.final_features)
        test_data["target_" + self.target_name] = features_class.calculate_target(self.target_name, self.target_window)

        test_data = self._clean_data(test_data)
        X_test = test_data[self.final_features]
        y_test = test_data["target_" + self.target_name]

        pred = self.rf.predict(X_test)
        
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        mape = mean_absolute_percentage_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        # relative data
        rel_mse = mse / np.var(y_test)

        self.metrics = {
            "mse" : mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape" : mape,
            "rel_mse" : rel_mse
        }
        return self.metrics

    def _clean_data(self, df):
        """
            returns cleaned df: 
                1. replace +/- infty with nan
                2. remove nan data
                3. clips outliers if winsorize set to true
        """

        # ensure properly sorted
        df = df.sort_values("Date")
        
        # replace infinities (if applicable)
        df = df.replace([np.inf, -np.inf], np.nan)

        # remove nan
        col_names = self.final_features + ["target_"+self.target_name]
        df = df.dropna(subset=col_names)

        # winsorize (if applicable)
        if self.winsorize:
            for col in self.features:
                lower =df[col].quantile(self.lower_q)
                upper = df[col].quantile(self.upper_q)

                df[col] = df[col].clip(lower, upper)
        return df

    