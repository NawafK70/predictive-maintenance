"""Train, compare, evaluate and save the machine-failure classifier.

Run:  python -m pdm.train            (default: threshold tuned for F2, recall-weighted)
      python -m pdm.train --beta 1   (balance precision and recall equally)

Workflow:
  1. Stratified hold-out test split (never touched until the final evaluation).
  2. 5-fold CV on the training split to compare candidate models by PR-AUC
     (failures are ~3% of rows, so accuracy and ROC-AUC are misleading).
  3. Out-of-fold probabilities pick the decision threshold that maximises F-beta.
  4. Refit the winner on the training split, score the test split once.
  5. Save pipeline + metadata + reference data for drift monitoring.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pdm import config
from pdm.data import download, load
from pdm.features import ENGINEERED, add_features


def build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
            ("num", StandardScaler(), config.NUMERIC + ENGINEERED),
        ]
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )


def candidates() -> dict:
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=rs,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            random_state=rs,
        ),
        # No scale_pos_weight: the tuned threshold handles the class imbalance,
        # and leaving it at 1 keeps the probabilities closer to real failure rates.
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=4,
            random_state=rs,
        ),
    }


def best_threshold(y_true, proba, beta: float) -> float:
    """Threshold that maximises F-beta. beta > 1 favours recall (fewer missed failures)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]
    denom = beta**2 * p + r
    fbeta = np.divide((1 + beta**2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beta", type=float, default=2.0, help="F-beta used to pick the threshold (default 2)")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    rs = config.RANDOM_STATE

    df = load(download())
    X, y, modes = df[config.FEATURES], df[config.TARGET], df[config.FAILURE_MODES]
    X_train, X_test, y_train, y_test, _, modes_test = train_test_split(
        X, y, modes, test_size=args.test_size, stratify=y, random_state=rs
    )
    print(f"Train {len(X_train):,} rows ({y_train.sum()} failures) | Test {len(X_test):,} rows ({y_test.sum()} failures)\n")

    # 1. Model selection by cross-validated PR-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    comparison = {}
    print(f"{'model':<24}{'CV PR-AUC':>16}{'CV ROC-AUC':>12}")
    for name, estimator in candidates().items():
        scores = cross_validate(
            build_pipeline(estimator), X_train, y_train, cv=cv,
            scoring=["average_precision", "roc_auc"], n_jobs=-1,
        )
        ap, auc = scores["test_average_precision"], scores["test_roc_auc"]
        comparison[name] = {
            "cv_average_precision": round(float(ap.mean()), 4),
            "cv_average_precision_std": round(float(ap.std()), 4),
            "cv_roc_auc": round(float(auc.mean()), 4),
        }
        print(f"{name:<24}{ap.mean():>9.4f} ± {ap.std():.3f}{auc.mean():>12.4f}")
    best_name = max(comparison, key=lambda n: comparison[n]["cv_average_precision"])
    print(f"\nSelected: {best_name}")

    # 2. Decision threshold from out-of-fold predictions (no test-set peeking)
    pipeline = build_pipeline(candidates()[best_name])
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = best_threshold(y_train, oof, args.beta)
    print(f"Threshold (max F{args.beta:g} on out-of-fold): {threshold:.4f}")

    # 3. Final fit + one-shot test evaluation
    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)

    flagged = test_proba >= threshold
    by_mode = {}
    for mode in config.FAILURE_MODES:
        mask = ((modes_test[mode] == 1) & (y_test == 1)).to_numpy()
        by_mode[mode] = {
            "name": config.FAILURE_MODE_NAMES[mode],
            "n_test_failures": int(mask.sum()),
            "recall": round(float(flagged[mask].mean()), 4) if mask.any() else None,
        }

    imp = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=rs, n_jobs=-1
    )
    importance = sorted(
        ({"feature": f, "importance": round(float(m), 4)} for f, m in zip(config.FEATURES, imp.importances_mean)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    print("\nHold-out test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:<18} {v}")
    print("\nRecall by failure mode (test):")
    for mode, info in by_mode.items():
        rec = "n/a" if info["recall"] is None else f"{info['recall']:.0%}"
        print(f"  {mode.upper()} {info['name']:<26} {rec:>5}  (n={info['n_test_failures']})")
    print("\nPermutation importance (drop in PR-AUC):")
    for d in importance:
        print(f"  {d['feature']:<24} {d['importance']:.4f}")

    # 4. Persist artefacts
    now = datetime.now(timezone.utc)
    metadata = {
        "model_version": now.strftime("%Y%m%d-%H%M%S"),
        "trained_at": now.isoformat(timespec="seconds"),
        "model_name": best_name,
        "threshold": threshold,
        "threshold_beta": args.beta,
        "features": config.FEATURES,
        "train_rows": len(X_train),
        "train_failure_rate": round(float(y_train.mean()), 4),
        "model_comparison": comparison,
        "test_metrics": test_metrics,
        "test_recall_by_failure_mode": by_mode,
        "permutation_importance": importance,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_DIR / config.MODEL_FILE)
    (config.MODEL_DIR / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    X_train.to_csv(config.MODEL_DIR / config.REFERENCE_FILE, index=False)
    print(f"\nSaved model {metadata['model_version']} to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
