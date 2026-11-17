"""Train and evaluate remaining-useful-life models on C-MAPSS.

Run:  python -m pdm.rul.train                 (FD001)
      python -m pdm.rul.train --subset FD004
      python -m pdm.rul.train --subset all     (all four + summary table)

Workflow per subset:
  1. Target = cycles until failure, capped at 125 (early-life RUL is unknowable).
  2. GroupKFold by engine: an engine's cycles never sit in both train and
     validation folds, otherwise the model memorises engines and CV lies.
  3. Compare models by CV RMSE, refit the best on all training engines.
  4. Score the official test set at each engine's last observed cycle:
     RMSE, MAE and the NASA PHM08 score (late predictions cost more than early ones).
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from pdm import config
from pdm.rul.data import load_test, load_train
from pdm.rul.features import RULFeaturizer


def build_pipeline(subset: str, estimator) -> Pipeline:
    return Pipeline([
        ("features", RULFeaturizer(n_conditions=config.CMAPSS_SUBSETS[subset])),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def candidates() -> dict:
    rs = config.RANDOM_STATE
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=rs),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, min_samples_leaf=40, l2_regularization=1.0, random_state=rs),
        "xgboost": XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, tree_method="hist", n_jobs=4, random_state=rs),
    }


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function. d > 0 means predicting failure too late (dangerous)."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def last_cycle_positions(df) -> np.ndarray:
    """Row positions of each engine's last cycle, ordered by unit number."""
    return df.reset_index(drop=True).groupby("unit")["cycle"].idxmax().sort_index().to_numpy()


def train_subset(subset: str) -> dict:
    print(f"\n=== {subset} ({config.CMAPSS_SUBSETS[subset]} operating condition(s)) ===")
    train = load_train(subset)
    X, y = train, train["rul"].clip(upper=config.RUL_CAP)
    groups = train["unit"]

    cv = GroupKFold(n_splits=5)
    comparison = {}
    print(f"{'model':<24}{'CV RMSE':>16}")
    for name, est in candidates().items():
        scores = cross_validate(build_pipeline(subset, est), X, y, groups=groups, cv=cv,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
        rmse = -scores["test_score"]
        comparison[name] = {"cv_rmse": round(float(rmse.mean()), 3), "cv_rmse_std": round(float(rmse.std()), 3)}
        print(f"{name:<24}{rmse.mean():>9.2f} ± {rmse.std():.2f}")
    best = min(comparison, key=lambda n: comparison[n]["cv_rmse"])
    print(f"Selected: {best}")

    pipeline = build_pipeline(subset, candidates()[best]).fit(X, y)

    test, true_rul = load_test(subset)
    pred = np.clip(pipeline.predict(test)[last_cycle_positions(test)], 0, config.RUL_CAP)
    err = pred - true_rul
    rmse = lambda e: round(float(np.sqrt(np.mean(e**2))), 2)  # noqa: E731
    near_failure = true_rul <= 50
    test_metrics = {
        "rmse": rmse(err),
        # Healthy engines with true RUL > cap can't be predicted above the cap by design;
        # FD002/FD004 have many of them, which dominates the uncapped RMSE.
        "rmse_vs_capped_truth": rmse(pred - np.minimum(true_rul, config.RUL_CAP)),
        # The number that matters for scheduling maintenance.
        "rmse_last_50_cycles": rmse(err[near_failure]),
        "pct_engines_above_cap": round(float(np.mean(true_rul > config.RUL_CAP) * 100), 1),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "nasa_score": round(nasa_score(true_rul, pred), 1),
        "late_predictions_pct": round(float(np.mean(err > 0) * 100), 1),
        "within_15_cycles_pct": round(float(np.mean(np.abs(err) <= 15) * 100), 1),
        "n_engines": int(len(true_rul)),
    }
    print("Test (last cycle per engine): " + ", ".join(f"{k}={v}" for k, v in test_metrics.items()))

    now = datetime.now(timezone.utc)
    featurizer = pipeline.named_steps["features"]
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "subset": subset,
        "n_conditions": config.CMAPSS_SUBSETS[subset],
        "rul_cap": config.RUL_CAP,
        "model_name": best,
        "sensors_used": featurizer.sensors_,
        "n_features": len(featurizer.feature_names_),
        "train_engines": int(train["unit"].nunique()),
        "model_comparison": comparison,
        "test_metrics": test_metrics,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    out = config.RUL_MODEL_DIR / subset
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / config.MODEL_FILE)
    (out / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    print(f"Saved to {out}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001", choices=[*config.CMAPSS_SUBSETS, "all"])
    args = parser.parse_args()
    subsets = list(config.CMAPSS_SUBSETS) if args.subset == "all" else [args.subset]

    results = [train_subset(s) for s in subsets]
    if len(results) > 1:
        print(f"\n{'subset':<7}{'conds':>6}{'model':>23}{'CV RMSE':>9}{'test RMSE':>10}"
              f"{'capped':>8}{'last 50':>9}{'NASA':>7}{'±15 cyc':>9}")
        for m in results:
            t = m["test_metrics"]
            print(f"{m['subset']:<7}{m['n_conditions']:>6}{m['model_name']:>23}"
                  f"{m['model_comparison'][m['model_name']]['cv_rmse']:>9.2f}{t['rmse']:>10.2f}"
                  f"{t['rmse_vs_capped_truth']:>8.2f}{t['rmse_last_50_cycles']:>9.2f}"
                  f"{t['nasa_score']:>7.0f}{t['within_15_cycles_pct']:>8.0f}%")


if __name__ == "__main__":
    main()
