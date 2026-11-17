"""Download, validate and load the AI4I 2020 predictive maintenance dataset.

Run:  python -m pdm.data
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from pdm import config

COLUMN_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def download(force: bool = False) -> Path:
    """Fetch the CSV from UCI unless it is already on disk."""
    csv_path = config.RAW_CSV
    if csv_path.exists() and not force:
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = csv_path.with_suffix(".zip")
    print(f"Downloading {config.DATASET_URL}")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open("ai4i2020.csv") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return csv_path


def load(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV, rename to snake_case and run basic schema checks."""
    df = pd.read_csv(path or config.RAW_CSV)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]

    if df[config.FEATURES + [config.TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values in feature/target columns")
    bad_types = set(df["type"].unique()) - {"L", "M", "H"}
    if bad_types:
        raise ValueError(f"Unexpected product types: {bad_types}")
    return df


def main() -> None:
    df = load(download())
    print(f"Rows: {len(df):,}")
    print(f"Failure rate: {df[config.TARGET].mean():.2%}  ({df[config.TARGET].sum()} failures)")
    print("Failure modes:")
    for mode in config.FAILURE_MODES:
        print(f"  {mode.upper()} {config.FAILURE_MODE_NAMES[mode]:<26} {df[mode].sum():>4}")
    print("\nFeature summary:")
    print(df[config.NUMERIC].describe().T[["mean", "std", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
