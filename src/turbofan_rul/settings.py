"""Paths and dataset constants. Experiment parameters live in configs/*.toml."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
# Sub-dataset -> number of operating conditions (flight regimes).
SUBSETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SUBSET_DESCRIPTIONS = {
    "FD001": "1 operating condition · 1 fault mode",
    "FD002": "6 operating conditions · 1 fault mode",
    "FD003": "1 operating condition · 2 fault modes",
    "FD004": "6 operating conditions · 2 fault modes",
}

# Environment overrides let the container (or a model-registry download) point elsewhere.
DATA_DIR = Path(os.getenv("RUL_DATA_DIR", ROOT / "data" / "raw" / "cmapss"))
MODEL_DIR = Path(os.getenv("RUL_MODEL_DIR", ROOT / "models"))  # one sub-folder per subset
REPORTS_DIR = ROOT / "reports"
DEFAULT_CONFIG = ROOT / "configs" / "default.toml"
PREDICTION_LOG = Path(os.getenv("RUL_PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))

# Artefacts inside models/<subset>/
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"  # sample of training readings, the drift baseline
