"""REST API: remaining useful life of a turbofan engine from its cycle history.

Run:   uvicorn turbofan_rul.api:app --reload      (docs at http://localhost:8000/docs)

Environment:
  RUL_MODEL_DIR        folder with one sub-folder per trained subset (default ./models)
  RUL_PREDICTION_LOG   JSONL file for request logging; set to "" to disable
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from turbofan_rul import __version__, settings
from turbofan_rul.data import COLUMNS, SENSORS, SETTINGS
from turbofan_rul.inference import RULPredictor, load_all


# --- schemas ------------------------------------------------------------------
class EngineCycle(BaseModel):
    cycle: int = Field(ge=1, description="Operating cycle number")
    settings: list[float] = Field(min_length=3, max_length=3, description="Operational settings 1-3")
    sensors: list[float] = Field(min_length=21, max_length=21, description="Sensor measurements 1-21")


class RULRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"cycles": [
        {"cycle": 1, "settings": [-0.0007, -0.0004, 100.0],
         "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30,
                     47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.00, 39.06, 23.4190]},
    ]}]})
    cycles: list[EngineCycle] = Field(
        min_length=1, max_length=2000,
        description="The engine's history so far, in any order. Rolling features need ~20+ cycles.")


class RULPrediction(BaseModel):
    rul_cycles: float = Field(description="Predicted operating cycles until failure")
    at_cap: bool = Field(description="No measurable degradation yet: RUL is at least the cap, possibly more")
    rul_cap: int
    cycles_seen: int
    subset: str
    model_version: str


# --- request logging (input to rul-monitor) -------------------------------------
class PredictionLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, pred: RULPrediction, latest: pd.Series) -> None:
        if not self.path:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "subset": pred.subset,
            "model_version": pred.model_version,
            "cycle": int(latest["cycle"]),
            **{c: float(latest[c]) for c in SETTINGS + SENSORS},
            "rul_cycles": pred.rul_cycles,
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# --- app ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("RUL_MODEL_DIR", settings.MODEL_DIR))
    app.state.models = load_all(model_dir)
    if not app.state.models:
        # Fail fast: a container without a model should crash, not serve errors.
        raise RuntimeError(f"No trained models under {model_dir}; run rul-train first")
    log_path = os.getenv("RUL_PREDICTION_LOG", str(settings.PREDICTION_LOG))
    app.state.logger = PredictionLogger(Path(log_path) if log_path else None)
    yield


app = FastAPI(
    title="Turbofan RUL API",
    description="Remaining useful life of turbofan engines from C-MAPSS-style cycle histories.",
    version=__version__,
    lifespan=lifespan,
)


def _models(request: Request) -> dict[str, RULPredictor]:
    return request.app.state.models


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "models": {name: m.version for name, m in _models(request).items()}}


@app.get("/models")
def models(request: Request) -> dict:
    keys = ("model_version", "trained_at", "model_name", "n_conditions", "rul_cap", "test_metrics")
    return {name: {k: m.metadata[k] for k in keys if k in m.metadata} for name, m in _models(request).items()}


@app.post("/predict/{subset}", response_model=RULPrediction)
def predict(subset: str, req: RULRequest, request: Request) -> RULPrediction:
    loaded = _models(request)
    if subset not in loaded:
        raise HTTPException(404, f"No model for '{subset}'. Available: {sorted(loaded)}")
    if len({c.cycle for c in req.cycles}) != len(req.cycles):
        raise HTTPException(422, "duplicate cycle numbers")

    model = loaded[subset]
    history = pd.DataFrame([[0, c.cycle, *c.settings, *c.sensors] for c in req.cycles], columns=COLUMNS)
    rul = model.predict_latest(history)
    pred = RULPrediction(
        rul_cycles=round(rul, 1),
        at_cap=model.is_at_cap(rul),
        rul_cap=model.rul_cap,
        cycles_seen=len(history),
        subset=model.subset,
        model_version=model.version,
    )
    request.app.state.logger.log(pred, history.loc[history["cycle"].idxmax()])
    return pred
