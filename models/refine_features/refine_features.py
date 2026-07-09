import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

class RefineFeatures:
    """
        Cleans input data
        Refines features to deal with correlated ones
    """

    def __init__(self, df, features : list, target : str, corr_threshold = 0.9, rf_refine=True):
        """
            Assume df has features calculated
            target is target value to estimate: should be full column name
            corr_threshold: float value used to determine which features are dropped in case of high correlation
            performs additional rf refinement if rf_refine is True
        """
        self.rf_refine = rf_refine
        self.features = features
        self.final_features = None
        self.target = target
        self.corr_threshold = corr_threshold
        self.df = df

    def _cluster_features(self):
        """
            Calculated features may be correlated
            Cluster accordingly using correlation threshold
            Returns cluster map
        """
        # weight features (maybe not?)
        df = self.df.copy()
        df[self.features] = df[self.features].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        corr = df[self.features].corr().abs()
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
        scores = self.df[self.features + [self.target]].corr()[self.target].abs()

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
        X = self.df[features]
        y = self.df[self.target]

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

    def refine_features(self):
        """
            Assumes have clean data
            Determines which features to keep
            Returns list of features to keep and df with only those features and target
        """

        cluster_map = self._cluster_features()
        selected = self._from_clusters(cluster_map)
        
        if self.rf_refine:
            selected = self._rf_refine(selected)

        self.final_features = selected

        return self.final_features, self.df[self.final_features + [self.target]]

    