"""Model registry and pipeline construction.

A pipeline is: RULFeaturizer -> StandardScaler -> regressor. Because feature
engineering lives inside the pipeline, the saved model takes raw engine cycles
and the API/demo can never compute features differently from training.
"""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from turbofan_rul import settings
from turbofan_rul.features import RULFeaturizer

# name -> (estimator class, fixed arguments not meant to be tuned in the config)
REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "ridge": (Ridge, {}),
    "random_forest": (RandomForestRegressor, {"n_jobs": -1}),
    "hist_gradient_boosting": (HistGradientBoostingRegressor, {}),
    "xgboost": (XGBRegressor, {"tree_method": "hist", "n_jobs": 4}),
}
SEEDED = {"random_forest", "hist_gradient_boosting", "xgboost"}


def make_estimator(name: str, params: dict[str, Any], seed: int):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    cls, fixed = REGISTRY[name]
    kwargs = {**fixed, **params}
    if name in SEEDED:
        kwargs["random_state"] = seed
    return cls(**kwargs)


def build_pipeline(subset: str, model_name: str, config: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("features", RULFeaturizer(n_conditions=settings.SUBSETS[subset], **config["features"])),
        ("scale", StandardScaler()),
        ("model", make_estimator(model_name, config["models"][model_name], config["seed"])),
    ])
