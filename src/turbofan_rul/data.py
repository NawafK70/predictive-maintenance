"""Download and load NASA C-MAPSS.

Each row is one engine at one operating cycle: unit, cycle, 3 operational settings
and 21 sensor readings. Training engines run until failure; test engines stop some
time before failure, and RUL_FDxxx.txt holds the true RUL at their last cycle.

CLI:  rul-data      (download if needed, then summarise the four subsets)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from turbofan_rul import settings

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Physical meaning of the sensors (Saxena et al., 2008).
SENSOR_NAMES = {
    "s1": "Fan inlet temperature", "s2": "LPC outlet temperature", "s3": "HPC outlet temperature",
    "s4": "LPT outlet temperature", "s5": "Fan inlet pressure", "s6": "Bypass-duct pressure",
    "s7": "HPC outlet pressure", "s8": "Physical fan speed", "s9": "Physical core speed",
    "s10": "Engine pressure ratio", "s11": "HPC outlet static pressure", "s12": "Fuel flow / Ps30 ratio",
    "s13": "Corrected fan speed", "s14": "Corrected core speed", "s15": "Bypass ratio",
    "s16": "Burner fuel-air ratio", "s17": "Bleed enthalpy", "s18": "Demanded fan speed",
    "s19": "Demanded corrected fan speed", "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}


def download(force: bool = False) -> Path:
    """Fetch the dataset from NASA's PHM data repository unless it is already on disk."""
    d = settings.DATA_DIR
    if (d / "train_FD001.txt").exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {settings.CMAPSS_URL}")
    with urllib.request.urlopen(settings.CMAPSS_URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
        for name in inner.namelist():
            if name.endswith(".txt"):
                (d / Path(name).name).write_bytes(inner.read(name))  # basename only: no path traversal
    return d


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    return df


def load_train(subset: str) -> pd.DataFrame:
    """Training engines with the true (uncapped) RUL at every cycle."""
    df = _read(download() / f"train_{subset}.txt")
    df["rul"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def load_test(subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Test engines (truncated histories) and the true RUL at each engine's last
    cycle, ordered by unit number."""
    d = download()
    df = _read(d / f"test_{subset}.txt")
    rul = pd.read_csv(d / f"RUL_{subset}.txt", header=None).iloc[:, 0].to_numpy()
    if len(rul) != df["unit"].nunique():
        raise ValueError(f"{subset}: {len(rul)} RUL values for {df['unit'].nunique()} test engines")
    return df, rul


def main() -> None:
    download()
    print(f"{'subset':<8}{'conditions':>11}{'train engines':>15}{'test engines':>14}"
          f"{'train rows':>12}{'life (cycles) min/median/max':>32}")
    for subset, n_cond in settings.SUBSETS.items():
        train = load_train(subset)
        test, _ = load_test(subset)
        life = train.groupby("unit")["cycle"].max()
        print(f"{subset:<8}{n_cond:>11}{train['unit'].nunique():>15}{test['unit'].nunique():>14}"
              f"{len(train):>12,}{f'{life.min()} / {int(life.median())} / {life.max()}':>32}")


if __name__ == "__main__":
    main()
