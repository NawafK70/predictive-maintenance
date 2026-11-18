"""Train and evaluate RUL models on C-MAPSS.

CLI:  rul-train                         (FD001, configs/default.toml)
      rul-train --subset all
      rul-train --subset FD004 --config configs/my_experiment.toml

Per subset:
  1. Target = cycles until failure, capped (configs: target.rul_cap).
  2. GroupKFold by engine: an engine's cycles never sit in both train and validation
     folds, otherwise the model memorises engines and CV is optimistic.
  3. Compare the configured candidate models by CV RMSE; refit the best on all
     training engines.
  4. Score NASA's official test set once, at each engine's last observed cycle.
  5. Save model.joblib, metadata.json (metrics + config + git commit) and
     reference.csv (drift baseline) to models/<subset>/.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GroupKFold, cross_validate

from turbofan_rul import settings
from turbofan_rul.data import SENSORS, SETTINGS, load_test, load_train
from turbofan_rul.evaluate import evaluate, last_cycle_positions
from turbofan_rul.inference import RULPredictor
from turbofan_rul.models import build_pipeline


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=settings.ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=settings.ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def compare_models(subset: str, train: pd.DataFrame, config: dict) -> dict[str, dict]:
    y = train["rul"].clip(upper=config["target"]["rul_cap"])
    cv = GroupKFold(n_splits=config["cv"]["n_splits"])
    results = {}
    print(f"{'model':<24}{'CV RMSE':>16}")
    for name in config["models"]:
        scores = cross_validate(build_pipeline(subset, name, config), train, y, groups=train["unit"],
                                cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        rmse = -scores["test_score"]
        results[name] = {"cv_rmse": round(float(rmse.mean()), 3), "cv_rmse_std": round(float(rmse.std()), 3)}
        print(f"{name:<24}{rmse.mean():>9.2f} ± {rmse.std():.2f}")
    return results


def train_subset(subset: str, config: dict, config_path: Path) -> dict:
    print(f"\n=== {subset} ({settings.SUBSET_DESCRIPTIONS[subset]}) ===")
    cap = config["target"]["rul_cap"]
    train = load_train(subset)

    comparison = compare_models(subset, train, config)
    best = min(comparison, key=lambda n: comparison[n]["cv_rmse"])
    print(f"Selected: {best}")
    pipeline = build_pipeline(subset, best, config).fit(train, train["rul"].clip(upper=cap))

    now = datetime.now(UTC)
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "subset": subset,
        "n_conditions": settings.SUBSETS[subset],
        "rul_cap": cap,
        "model_name": best,
        "model_comparison": comparison,
        "train_engines": int(train["unit"].nunique()),
        "config_file": config_path.name,
        "config": config,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__},
    }
    predictor = RULPredictor(pipeline, metadata)
    metadata["sensors_used"] = predictor.featurizer.sensors_
    metadata["n_features"] = len(predictor.featurizer.feature_names_)

    test, true_rul = load_test(subset)
    pred = predictor.predict(test)[last_cycle_positions(test)]
    metadata["test_metrics"] = evaluate(true_rul, pred, cap)
    print("Test (last cycle per engine): "
          + ", ".join(f"{k}={v}" for k, v in metadata["test_metrics"].items()))

    out = settings.MODEL_DIR / subset
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / settings.MODEL_FILE)
    (out / settings.METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    save_reference(train, out, config["seed"])
    print(f"Saved to {out}")
    return metadata


def save_reference(train: pd.DataFrame, out: Path, seed: int) -> None:
    """Sample of training readings: the baseline that rul-monitor compares live traffic against."""
    sample = train[SETTINGS + SENSORS].sample(min(5000, len(train)), random_state=seed)
    sample.to_csv(out / settings.REFERENCE_FILE, index=False)


def print_summary(results: list[dict]) -> None:
    print(f"\n{'subset':<7}{'model':>16}{'CV RMSE':>9}{'test RMSE':>10}{'capped':>8}"
          f"{'last 50':>9}{'NASA':>7}{'±15 cyc':>9}")
    for m in results:
        t = m["test_metrics"]
        print(f"{m['subset']:<7}{m['model_name']:>16}{m['model_comparison'][m['model_name']]['cv_rmse']:>9.2f}"
              f"{t['rmse']:>10.2f}{t['rmse_vs_capped_truth']:>8.2f}{t['rmse_last_50_cycles']:>9.2f}"
              f"{t['nasa_score']:>7.0f}{t['within_15_cycles_pct']:>8.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001", choices=[*settings.SUBSETS, "all"])
    parser.add_argument("--config", type=Path, default=settings.DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    subsets = list(settings.SUBSETS) if args.subset == "all" else [args.subset]
    results = [train_subset(s, config, args.config) for s in subsets]
    print_summary(results)


if __name__ == "__main__":
    main()
