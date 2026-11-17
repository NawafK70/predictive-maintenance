"""Physics-informed features.

This function runs *inside* the saved sklearn pipeline, so training and the API
compute features with the same code (no train/serve skew).
"""
from __future__ import annotations

import math

import pandas as pd

ENGINEERED = ["temp_diff_k", "power_w", "strain_min_nm"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Heat dissipation failures happen when process and air temperature are too close.
    out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    # Mechanical power P = torque * angular velocity; power failures sit at the extremes.
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * math.pi / 60
    # Overstrain failures depend on tool wear x torque.
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out
