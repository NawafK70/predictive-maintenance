"""Paths and constants shared by training, serving and monitoring."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
CMAPSS_DIR = ROOT / "data" / "raw" / "cmapss"
# Sub-dataset -> number of operating conditions (flight regimes).
CMAPSS_SUBSETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

# One folder per subset: models/FD001/{model.joblib, metadata.json, reference.csv}.
# Overridable so the container (or a model-registry download) can point elsewhere.
MODEL_DIR = Path(os.getenv("PDM_MODEL_DIR", ROOT / "models"))
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"  # sample of training readings, the drift baseline

PREDICTION_LOG = ROOT / "logs" / "predictions.jsonl"

RANDOM_STATE = 42

# Early in life an engine shows no degradation, so its true RUL can't be inferred.
# Standard practice: cap the target ("piecewise-linear RUL") at 125 cycles.
RUL_CAP = 125
