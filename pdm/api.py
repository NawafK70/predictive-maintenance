"""REST API that predicts turbofan remaining useful life from an engine's history.

Run locally:  uvicorn pdm.api:app --reload
Docs:         http://localhost:8000/docs

Environment:
  PDM_MODEL_DIR        folder with one sub-folder per trained subset (default ./models)
  PDM_PREDICTION_LOG   JSONL file for request logging; set to "" to disable
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from pdm import config
from pdm.data import COLUMNS, SENSORS, SETTINGS


class EngineCycle(BaseModel):
    cycle: int = Field(ge=1, description="Operating cycle number")
    settings: list[float] = Field(min_length=3, max_length=3, description="Operational settings 1-3")
    sensors: list[float] = Field(min_length=21, max_length=21, description="Sensor measurements 1-21")


class RULRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"cycles": [
            {"cycle": 1, "settings": [-0.0007, -0.0004, 100.0],
             "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30,
                         47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.00, 39.06, 23.4190]},
        ]}]}
    )
    cycles: list[EngineCycle] = Field(
        min_length=1, max_length=2000,
        description="The engine's history so far. Rolling features need ~20+ cycles to be reliable.")


class RULPrediction(BaseModel):
    rul_cycles: float = Field(description="Predicted operating cycles until failure")
    at_cap: bool = Field(description="True if the engine shows no measurable degradation yet "
                                     "(RUL is at least the cap, possibly much more)")
    rul_cap: int
    cycles_seen: int
    subset: str
    model_version: str


class RULModel:
    def __init__(self, subset_dir: Path):
        self.pipeline = joblib.load(subset_dir / config.MODEL_FILE)
        self.metadata = json.loads((subset_dir / config.METADATA_FILE).read_text())
        self.subset = self.metadata["subset"]
        self.version = self.metadata["model_version"]
        self.cap = int(self.metadata["rul_cap"])

    def predict(self, req: RULRequest) -> tuple[RULPrediction, pd.DataFrame]:
        cycles = sorted(req.cycles, key=lambda c: c.cycle)
        if len({c.cycle for c in cycles}) != len(cycles):
            raise HTTPException(422, "duplicate cycle numbers")
        df = pd.DataFrame([[0, c.cycle, *c.settings, *c.sensors] for c in cycles], columns=COLUMNS)
        rul = float(np.clip(self.pipeline.predict(df)[-1], 0, self.cap))  # rows come back in cycle order
        pred = RULPrediction(
            rul_cycles=round(rul, 1),
            at_cap=rul >= self.cap - 0.5,
            rul_cap=self.cap,
            cycles_seen=len(cycles),
            subset=self.subset,
            model_version=self.version,
        )
        return pred, df


class PredictionLogger:
    """Appends the latest reading + prediction of each request, for drift checks (pdm.monitor)."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, pred: RULPrediction, history: pd.DataFrame) -> None:
        if not self.path:
            return
        last = history.iloc[-1]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subset": pred.subset,
            "model_version": pred.model_version,
            "cycle": int(last["cycle"]),
            **{c: float(last[c]) for c in SETTINGS + SENSORS},
            "rul_cycles": pred.rul_cycles,
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def load_models(root: Path) -> dict[str, RULModel]:
    models = {d.name: RULModel(d) for d in sorted(root.iterdir())
              if (d / config.MODEL_FILE).exists()} if root.exists() else {}
    if not models:
        # Fail fast: a container without a model should crash, not serve errors.
        raise RuntimeError(f"No trained models under {root}; run python -m pdm.train")
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models(Path(os.getenv("PDM_MODEL_DIR", config.MODEL_DIR)))
    log_env = os.getenv("PDM_PREDICTION_LOG", str(config.PREDICTION_LOG))
    app.state.logger = PredictionLogger(Path(log_env) if log_env else None)
    yield


app = FastAPI(
    title="Turbofan RUL API",
    description="Remaining useful life of turbofan engines from C-MAPSS-style cycle histories.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "models": {n: m.version for n, m in request.app.state.models.items()}}


@app.get("/models")
def models(request: Request) -> dict:
    return {name: {k: m.metadata[k] for k in ("model_version", "trained_at", "model_name", "n_conditions",
                                             "rul_cap", "test_metrics") if k in m.metadata}
            for name, m in request.app.state.models.items()}


@app.post("/predict/{subset}", response_model=RULPrediction)
def predict(subset: str, req: RULRequest, request: Request) -> RULPrediction:
    loaded = request.app.state.models
    if subset not in loaded:
        raise HTTPException(404, f"No model for '{subset}'. Available: {sorted(loaded)}")
    pred, history = loaded[subset].predict(req)
    request.app.state.logger.log(pred, history)
    return pred
