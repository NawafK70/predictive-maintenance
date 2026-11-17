"""Paths, column names and constants shared by training, serving and monitoring."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
RAW_CSV = ROOT / "data" / "raw" / "ai4i2020.csv"

# Overridable so the container (or a model registry download) can point elsewhere.
MODEL_DIR = Path(os.getenv("PDM_MODEL_DIR", ROOT / "models"))
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"  # training inputs, used as the drift baseline

PREDICTION_LOG = ROOT / "logs" / "predictions.jsonl"

RANDOM_STATE = 42

# Model inputs: exactly what a sensor/PLC feed would send to the API.
CATEGORICAL = ["type"]
NUMERIC = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
FEATURES = CATEGORICAL + NUMERIC

TARGET = "machine_failure"
# Failure-mode flags. They describe *how* a machine failed, so they leak the
# target and must never be used as features. Kept only for evaluation.
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
# --- NASA C-MAPSS turbofan: remaining useful life (RUL) -------------------------
CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
CMAPSS_DIR = ROOT / "data" / "raw" / "cmapss"
# Sub-dataset -> number of operating conditions (flight regimes).
CMAPSS_SUBSETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
RUL_MODEL_DIR = MODEL_DIR / "rul"
# Early in life an engine shows no degradation, so its true RUL can't be inferred.
# Standard practice: cap the target ("piecewise-linear RUL") at 125 cycles.
RUL_CAP = 125

FAILURE_MODE_NAMES = {
    "twf": "tool wear failure",
    "hdf": "heat dissipation failure",
    "pwf": "power failure",
    "osf": "overstrain failure",
    "rnf": "random failure",
}
