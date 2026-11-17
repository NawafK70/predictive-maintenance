"""Data-drift check: compare logged API inputs against the training data.

Run:  python -m pdm.monitor [--log logs/predictions.jsonl] [--min-rows 200]

Uses the Population Stability Index (PSI) per feature:
  < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 significant drift -> retrain
Exit code 1 when any feature drifts, so a scheduler/CI job can trigger retraining.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pdm import config

MODERATE, SIGNIFICANT = 0.10, 0.25


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    return _psi(e, a)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected) | set(actual))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(e, a)


def _psi(e: np.ndarray, a: np.ndarray) -> float:
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.FEATURES:
        if col in config.CATEGORICAL:
            value = psi_categorical(reference[col], current[col])
        else:
            value = psi_numeric(reference[col].to_numpy(), current[col].to_numpy())
        status = "DRIFT" if value > SIGNIFICANT else "moderate" if value > MODERATE else "stable"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=config.PREDICTION_LOG)
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No prediction log at {args.log}")
        return 0
    current = pd.read_json(args.log, lines=True)
    if len(current) < args.min_rows:
        print(f"Only {len(current)} logged predictions; need {args.min_rows} for a reliable check.")
        return 0
    reference = pd.read_csv(config.MODEL_DIR / config.REFERENCE_FILE)

    report = drift_report(reference, current)
    print(f"Reference: {len(reference):,} training rows | Current: {len(current):,} logged predictions\n")
    print(report.to_string(index=False))
    print(f"\nMean predicted failure probability: {current['failure_probability'].mean():.2%}")

    drifted = report.loc[report["status"] == "DRIFT", "feature"].tolist()
    if drifted:
        print(f"\nDrift detected in {drifted} -> investigate and retrain (python -m pdm.train).")
        return 1
    print("\nNo significant drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
