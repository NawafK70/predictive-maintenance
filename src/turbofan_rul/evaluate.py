"""Evaluation metrics for RUL prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEAR_FAILURE = 50  # cycles: the window in which maintenance actually gets scheduled


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 scoring function (lower is better). Exponential in the error, and harsher
    on late predictions (d > 0: the engine fails before the model said it would)."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def last_cycle_positions(df: pd.DataFrame) -> np.ndarray:
    """Row positions of each engine's last observed cycle, ordered by unit number."""
    return df.reset_index(drop=True).groupby("unit")["cycle"].idxmax().sort_index().to_numpy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, rul_cap: int) -> dict:
    """Metrics at each test engine's last observed cycle.

    `rmse` is the standard benchmark number. `rmse_vs_capped_truth` excludes the error
    the model cannot avoid by design (healthy engines with true RUL above the cap), and
    `rmse_last_50_cycles` is the accuracy that matters for scheduling maintenance.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    near = y_true <= NEAR_FAILURE
    return {
        "rmse": round(rmse(err), 2),
        "rmse_vs_capped_truth": round(rmse(y_pred - np.minimum(y_true, rul_cap)), 2),
        "rmse_last_50_cycles": round(rmse(err[near]), 2) if near.any() else None,
        "mae": round(float(np.mean(np.abs(err))), 2),
        "nasa_score": round(nasa_score(y_true, y_pred), 1),
        "late_predictions_pct": round(float(np.mean(err > 0) * 100), 1),
        "within_15_cycles_pct": round(float(np.mean(np.abs(err) <= 15) * 100), 1),
        "pct_engines_above_cap": round(float(np.mean(y_true > rul_cap) * 100), 1),
        "n_engines": int(len(y_true)),
    }
