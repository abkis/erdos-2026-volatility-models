import yfinance as yf
import seaborn as sns

from typing import List

class RF_Model:
    """
        Handles logic of training/testing/evaluating/etc RF model
        Handles data download and preprocessing
        Deals with daily historical data
    """

    def __init__(self, stocks : List[str], window : int, start_date : str, end_date : str, features : List[str]| None, target : str,
                lookback = 22, lamb = 0.94, winsorize = True, corr_threshold = 0.9, lower_q=0.01, upper_q=0.99, rf_refine=True):
        """
            Initialize class
            stocks: list of stock names
            window: positive integer for rolling window
            start_date, end_date: start/end date for stock info, format yyyy-mm-dd
            features: list of strings, must be from ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
                If none defaults to full list
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
            winsorize: boolean that determines if outliers are clipped
            corr_threshold: float value used to determine which features are dropped in case of high correlation
            lower_q, upper_q are quantiles for winsorization
            performs additional rf refinement if rf_refine is True
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
        self.all_features = ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
        if features is not None:
            for feature in features:
                assert feature in self.all_features
            self.features = features
        else:
            self.features = self.all_features

        self.target = target # DV to learn
        
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
        # ensure properly sorted
        self.stock_df = self.stock_df.sort_values(["Symbol", "Date"])

        self.corr = None
        self.final_features = self.features 
        self.final_df = self.stock_df.copy()

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
        self.final_df[self.features] = feature_class.calculate_features(self.features)
        self.final_df[self.target] = feature_class.calculate_target(self.target)

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

    