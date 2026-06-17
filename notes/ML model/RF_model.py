import yfinance as yf
import seaborn as sns

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

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

class RF_Model:
    """
        Handles logic of training/testing/evaluating/etc RF model
        Handles data download and preprocessing
        Deals with daily historical data
    """

    def __init__(self, stocks : List[str], window : int, start_date : str, end_date : str, features : List[str]| None, target : str,
                winsorize = True, corr_threshold = 0.9, lookback = 22, lamb = 0.94, lower_q=0.01, upper_q=0.99, rf_refine=True):
        """
            Initialize class
            stocks: list of stock names
            window: positive integer for rolling window
            start_date, end_date: start/end date for stock info, format yyyy-mm-dd
            features: list of strings, must be from ["rolling_vol", "parkinson_vol", "close_close_vol", "GK_vol", "GKYZ", "vix_fix", "ewma", "macd", 
                "macd_signal", "macd_hist", "rsi", "adx", "atr", "bb_width", "obv", "cmf"]
                If none defaults to full list
            winsorize: boolean that determines if outliers are clipped
            corr_threshold: float value used to determine which features are dropped in case of high correlation
            lookback used in VIX proxy, usually 22
            lamb is a float between zero and one, usually 0.94 for daily data
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

    def visualize_corr(self):
        """
            use plt/sns to visualize correlation matrix
            assumes self.corr calculated/called after features are created
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.corr, cmap="coolwarm", center=0)
        plt.show()

    def _clean_data(self):
        """
            Assumes that features have been calculated
            Cleans data: 
                1. replace +/- infty with nan
                2. remove nan data
                3. clips outliers if winsorize set to true
        """

        # ensure properly sorted
        self.stock_df = self.stock_df.sort_values(["Symbol", "Date"])
        
        # replace infinities (if applicable)
        self.stock_df = self.stock_df.replace([np.inf, -np.inf], np.nan)

        # remove nan
        self.stock_df = self.stock_df.dropna()

        # winsorize (if applicable)
        if winsorize:
            for col in self.features:
                lower =self.stock_df[col].quantile(self.lower_q)
                upper = self.stock_df[col].quantile(self.upper_q)

                self.stock_df[col] = self.stock_df[col].clip(lower, upper)

    def _cluster_features(self):
        """
            Calculated features may be correlated
            Cluster accordingly using correlation threshold
            Returns cluster map
        """
        corr = self.corr.abs()
        dist = 1-corr
        np.fill_diagonal(dist.values, 0)

        Z = linkage(squareform(dist), method="average")

        # get clusters based on corr threshold
        clusters = fcluster(
            Z,
            t=1 - self.corr_threshold,
            criterion="distance"
        )

        cluster_map = pd.DataFrame({
            "feature": self.features,
            "cluster": clusters
        })

        return cluster_map

    def _from_clusters(self, cluster_map):
        """
            Input cluster_map from result of _cluster_features
            selects most important features depending on what is most correlated with target
        """
        scores = self.stocks_df[self.features + [self.target]].corr()[self.target].abs()

        selected = []

        for c in cluster_map["cluster"].unique():
            group = cluster_map[cluster_map["cluster"] == c]["feature"]

            best = group.loc[group.map(scores).idxmax()]
            selected.append(best)

        return selected

    def _rf_refine(self, features):
        """
            optional method
            refines features using rf
        """
        X = self.stock_df[features]
        y = self.stock_df[self.target]

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X, y)

        imp = permutation_importance(
            rf, X, y,
            n_repeats=5,
            random_state=42
        )

        imp_series = pd.Series(
            imp.importances_mean,
            index=features
        ).sort_values(ascending=False)

        # keep non-trivial features
        return imp_series[imp_series > 0.001].index.tolist()

    def fit(self):
        """
            fits random forest to data
            1. Calculates features
            2. Cleans results
            3. Deals with correlated data
        """
        self._calculate_fearures()
        self._clean_data()

        cluster_map = self._cluster_features()
        selected = self._from_clusters(cluster_map)
        
        if self.rf_refine:
            selected = self._rf_refine(selected)

        self.final_features = selected