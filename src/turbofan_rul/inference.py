"""Load trained models and predict RUL. Shared by the API, the demo and the report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from turbofan_rul import settings
from turbofan_rul.features import RULFeaturizer


@dataclass
class RULPredictor:
    pipeline: Pipeline
    metadata: dict[str, Any]

    @classmethod
    def load(cls, model_dir: Path) -> RULPredictor:
        return cls(
            pipeline=joblib.load(model_dir / settings.MODEL_FILE),
            metadata=json.loads((model_dir / settings.METADATA_FILE).read_text()),
        )

    @property
    def subset(self) -> str:
        return self.metadata["subset"]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    @property
    def rul_cap(self) -> int:
        return int(self.metadata["rul_cap"])

    @property
    def featurizer(self) -> RULFeaturizer:
        return self.pipeline.named_steps["features"]

    def predict(self, cycles: pd.DataFrame) -> np.ndarray:
        """RUL at every row (input row order), clipped to [0, cap]. Features are causal,
        so each row's prediction uses only that engine's earlier cycles."""
        return np.clip(self.pipeline.predict(cycles), 0, self.rul_cap)

    def predict_latest(self, history: pd.DataFrame) -> float:
        """RUL after the most recent cycle of a single engine's history."""
        if history["unit"].nunique() != 1:
            raise ValueError("history must contain exactly one engine")
        latest = int(np.argmax(history["cycle"].to_numpy()))
        return float(self.predict(history)[latest])

    def is_at_cap(self, rul: float) -> bool:
        """True when the engine shows no measurable degradation yet."""
        return rul >= self.rul_cap - 0.5


def engine_trajectory(model: RULPredictor, engine: pd.DataFrame, final_rul: float | None = None) -> pd.DataFrame:
    """Predicted RUL at every observed cycle of one engine, plus the actual RUL when the
    engine's RUL after its last cycle is known (as for NASA's test engines)."""
    engine = engine.sort_values("cycle")
    cycles = engine["cycle"].to_numpy()
    out = pd.DataFrame({"cycle": cycles, "predicted": model.predict(engine).round(1)})
    if final_rul is not None:
        out["actual"] = final_rul + cycles.max() - cycles
    return out


def available_models(root: Path = settings.MODEL_DIR) -> list[str]:
    return sorted(d.name for d in root.iterdir() if (d / settings.MODEL_FILE).exists()) if root.exists() else []


def load_all(root: Path = settings.MODEL_DIR) -> dict[str, RULPredictor]:
    return {name: RULPredictor.load(root / name) for name in available_models(root)}
