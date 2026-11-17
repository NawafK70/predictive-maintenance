"""Data-drift check: compare logged API readings against the training data.

Run:  python -m pdm.monitor --subset FD001 [--log logs/predictions.jsonl] [--min-rows 200]

Uses the Population Stability Index (PSI) per setting/sensor:
  < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 significant drift -> investigate/retrain
Exit code 1 when anything drifts, so a scheduler/CI job can trigger retraining.

Caveat: degradation itself shifts sensor values. A fleet that is mostly near
end-of-life will look "drifted" against training data that covers whole lives,
so read drift alongside the distribution of predicted RUL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pdm import config

MODERATE, SIGNIFICANT = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant in training: any change is drift
        return 0.0 if np.allclose(actual, expected[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        value = psi(reference[col].to_numpy(), current[col].to_numpy())
        status = "DRIFT" if value > SIGNIFICANT else "moderate" if value > MODERATE else "stable"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001", choices=list(config.CMAPSS_SUBSETS))
    parser.add_argument("--log", type=Path, default=config.PREDICTION_LOG)
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No prediction log at {args.log}")
        return 0
    current = pd.read_json(args.log, lines=True)
    current = current[current["subset"] == args.subset]
    if len(current) < args.min_rows:
        print(f"Only {len(current)} logged predictions for {args.subset}; need {args.min_rows}.")
        return 0

    model_dir = config.MODEL_DIR / args.subset
    reference = pd.read_csv(model_dir / config.REFERENCE_FILE)
    meta = json.loads((model_dir / config.METADATA_FILE).read_text())
    columns = ["setting_1", "setting_2", "setting_3", *meta["sensors_used"]]

    report = drift_report(reference, current, columns)
    print(f"{args.subset} | reference: {len(reference):,} training cycles | current: {len(current):,} logged\n")
    print(report.to_string(index=False))
    print(f"\nPredicted RUL: median {current['rul_cycles'].median():.0f} cycles, "
          f"{(current['rul_cycles'] <= 30).mean():.0%} of requests at <= 30 cycles")

    drifted = report.loc[report["status"] == "DRIFT", "feature"].tolist()
    if drifted:
        print(f"\nDrift detected in {drifted} -> investigate and retrain (python -m pdm.train).")
        return 1
    print("\nNo significant drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
