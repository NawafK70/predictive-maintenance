"""Time-series features for RUL, packaged as an sklearn transformer.

Steps (all fitted on training data only, and run inside the saved pipeline so
the API computes features exactly as training did):

1. Operating-condition normalisation. In FD002/FD004 each cycle is flown in one of
   six regimes (altitude / Mach / throttle). Raw sensor values jump between
   regimes far more than they move with degradation, so each sensor is
   standardised using the mean/std *of its regime* (regimes found by KMeans on
   the 3 settings).
2. Drop sensors that are constant (no information).
3. Per-engine causal rolling features: short and long rolling means (denoised
   level) and a rolling linear slope (degradation trend). Only past cycles are
   used, so a row's features never depend on the future.
4. Engine age (cycle number).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from turbofan_rul.data import SENSORS, SETTINGS

REQUIRED = ["unit", "cycle", *SETTINGS, *SENSORS]


class RULFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_conditions: int = 1, short_window: int = 5, long_window: int = 20,
                 slope_window: int = 20):
        self.n_conditions = n_conditions
        self.short_window = short_window
        self.long_window = long_window
        self.slope_window = slope_window

    # -- fitting --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        X = self._check(X)
        if self.n_conditions > 1:
            self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=0).fit(X[SETTINGS].to_numpy())
        cond = self.conditions(X)
        grouped = X[SENSORS].groupby(cond)
        mean, std = grouped.mean(), grouped.std().fillna(0.0)
        self.sensors_ = [s for s in SENSORS if (std[s] > 1e-6).any()]
        self.mean_ = mean[self.sensors_]
        self.std_ = std[self.sensors_].where(std[self.sensors_] > 1e-6, 1.0)
        self.feature_names_ = (
            ["cycle"]
            + [f"{s}_mean{self.short_window}" for s in self.sensors_]
            + [f"{s}_mean{self.long_window}" for s in self.sensors_]
            + [f"{s}_slope{self.slope_window}" for s in self.sensors_]
        )
        return self

    def conditions(self, X: pd.DataFrame) -> np.ndarray:
        """Operating-condition (regime) id of each row."""
        if self.n_conditions == 1:
            return np.zeros(len(X), dtype=int)
        return self.kmeans_.predict(X[SETTINGS].to_numpy())

    def normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Sensors standardised within their operating condition (same row order as X)."""
        cond = self.conditions(X)
        z = (X[self.sensors_].to_numpy() - self.mean_.loc[cond].to_numpy()) / self.std_.loc[cond].to_numpy()
        return pd.DataFrame(z, columns=self.sensors_, index=X.index)

    # -- transforming ---------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check(X).reset_index(drop=True)
        order = X.sort_values(["unit", "cycle"], kind="stable").index
        Xs = X.loc[order].reset_index(drop=True)

        z = self.normalize(Xs)
        units = Xs["unit"].to_numpy()
        t = Xs["cycle"].astype(float)

        def rmean(df: pd.DataFrame, w: int) -> pd.DataFrame:
            return df.groupby(units).rolling(w, min_periods=1).mean().droplevel(0).sort_index()

        short = rmean(z, self.short_window).add_suffix(f"_mean{self.short_window}")
        long = rmean(z, self.long_window).add_suffix(f"_mean{self.long_window}")

        # Rolling least-squares slope: cov(x, t) / var(t) over the window.
        w = self.slope_window
        m = rmean(pd.concat([z, z.mul(t, axis=0).add_suffix("_xt"),
                             pd.DataFrame({"_t": t, "_tt": t * t})], axis=1), w)
        var_t = (m["_tt"] - m["_t"] ** 2).to_numpy()[:, None]
        cov = m[[f"{s}_xt" for s in self.sensors_]].to_numpy() - m[self.sensors_].to_numpy() * m[["_t"]].to_numpy()
        slope = np.divide(cov, var_t, out=np.zeros_like(cov), where=var_t > 1e-9)
        slope = pd.DataFrame(slope, columns=[f"{s}_slope{w}" for s in self.sensors_])

        feats = pd.concat([Xs[["cycle"]].astype(float), short, long, slope], axis=1)
        feats.index = order  # back to the caller's row order
        return feats.sort_index()[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    @staticmethod
    def _check(X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return X
