"""REST API that scores machine sensor readings for failure risk.

Run locally:  uvicorn pdm.api:app --reload
Docs:         http://localhost:8000/docs

Environment:
  PDM_MODEL_DIR        folder with model.joblib + metadata.json (default ./models)
  PDM_PREDICTION_LOG   JSONL file for request logging; set to "" to disable
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from pdm import config
from pdm.rul.data import COLUMNS as RUL_COLUMNS


class MachineReading(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "type": "L",
                    "air_temperature_k": 302.5,
                    "process_temperature_k": 310.9,
                    "rotational_speed_rpm": 1320,
                    "torque_nm": 62.0,
                    "tool_wear_min": 210,
                }
            ]
        }
    )

    type: Literal["L", "M", "H"] = Field(description="Product quality variant: L(ow), M(edium), H(igh)")
    air_temperature_k: float = Field(gt=250, lt=350, description="Ambient air temperature [K]")
    process_temperature_k: float = Field(gt=250, lt=400, description="Process temperature [K]")
    rotational_speed_rpm: float = Field(gt=0, lt=10_000, description="Spindle speed [rpm]")
    torque_nm: float = Field(ge=0, lt=200, description="Torque [Nm]")
    tool_wear_min: float = Field(ge=0, lt=1_000, description="Accumulated tool wear [min]")


class BatchRequest(BaseModel):
    readings: list[MachineReading] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    failure_probability: float
    predicted_failure: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_version: str


class ModelService:
    def __init__(self, model_dir: Path, log_path: Path | None):
        self.pipeline = joblib.load(model_dir / config.MODEL_FILE)
        self.metadata = json.loads((model_dir / config.METADATA_FILE).read_text())
        self.threshold = float(self.metadata["threshold"])
        self.version = self.metadata["model_version"]
        self.log_path = log_path
        self._lock = threading.Lock()
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "high"
        if p >= self.threshold / 2:
            return "medium"
        return "low"

    def predict(self, readings: list[MachineReading]) -> list[Prediction]:
        rows = [r.model_dump() for r in readings]
        proba = self.pipeline.predict_proba(pd.DataFrame(rows)[config.FEATURES])[:, 1]
        preds = [
            Prediction(
                failure_probability=round(float(p), 4),
                predicted_failure=bool(p >= self.threshold),
                risk_level=self.risk_level(float(p)),
                threshold=round(self.threshold, 4),
                model_version=self.version,
            )
            for p in proba
        ]
        self._log(rows, preds)
        return preds

    def _log(self, rows: list[dict], preds: list[Prediction]) -> None:
        """Append inputs + outputs so drift can be measured later (pdm.monitor)."""
        if not self.log_path:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        lines = [
            json.dumps({"ts": ts, "model_version": self.version, **row,
                        "failure_probability": pred.failure_probability})
            for row, pred in zip(rows, preds)
        ]
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# --- Remaining useful life (C-MAPSS turbofan) -----------------------------------
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


class RULService:
    def __init__(self, subset_dir: Path):
        self.pipeline = joblib.load(subset_dir / config.MODEL_FILE)
        self.metadata = json.loads((subset_dir / config.METADATA_FILE).read_text())
        self.cap = int(self.metadata["rul_cap"])

    def predict(self, req: RULRequest) -> RULPrediction:
        cycles = sorted(req.cycles, key=lambda c: c.cycle)
        if len({c.cycle for c in cycles}) != len(cycles):
            raise HTTPException(422, "duplicate cycle numbers")
        df = pd.DataFrame(
            [[0, c.cycle, *c.settings, *c.sensors] for c in cycles], columns=RUL_COLUMNS)
        rul = float(np.clip(self.pipeline.predict(df)[-1], 0, self.cap))  # rows come back in cycle order
        return RULPrediction(
            rul_cycles=round(rul, 1),
            at_cap=rul >= self.cap - 0.5,
            rul_cap=self.cap,
            cycles_seen=len(cycles),
            subset=self.metadata["subset"],
            model_version=self.metadata["model_version"],
        )


def load_rul_services(root: Path) -> dict[str, RULService]:
    """Load every trained C-MAPSS subset model found under models/rul/ (optional)."""
    if not root.exists():
        return {}
    return {d.name: RULService(d) for d in sorted(root.iterdir()) if (d / config.MODEL_FILE).exists()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("PDM_MODEL_DIR", config.MODEL_DIR))
    log_env = os.getenv("PDM_PREDICTION_LOG", str(config.PREDICTION_LOG))
    # Fail fast: a container without a model should crash, not serve errors.
    app.state.service = ModelService(model_dir, Path(log_env) if log_env else None)
    app.state.rul = load_rul_services(model_dir / "rul")
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure risk from AI4I-style sensor readings (/predict) and turbofan "
                "remaining useful life from C-MAPSS-style engine histories (/rul/{subset}/predict).",
    version="1.0.0",
    lifespan=lifespan,
)


def _service(request: Request) -> ModelService:
    return request.app.state.service


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": _service(request).version}


@app.get("/model")
def model_info(request: Request) -> dict:
    meta = _service(request).metadata
    keys = ["model_version", "trained_at", "model_name", "threshold", "threshold_beta",
            "test_metrics", "test_recall_by_failure_mode", "permutation_importance"]
    return {k: meta[k] for k in keys if k in meta}


@app.post("/predict", response_model=Prediction)
def predict(reading: MachineReading, request: Request) -> Prediction:
    return _service(request).predict([reading])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest, request: Request) -> list[Prediction]:
    return _service(request).predict(batch.readings)


@app.get("/rul/models")
def rul_models(request: Request) -> dict:
    return {name: {k: s.metadata[k] for k in ("model_version", "model_name", "n_conditions", "test_metrics")}
            for name, s in request.app.state.rul.items()}


@app.post("/rul/{subset}/predict", response_model=RULPrediction)
def predict_rul(subset: str, req: RULRequest, request: Request) -> RULPrediction:
    services = request.app.state.rul
    if subset not in services:
        raise HTTPException(404, f"No RUL model for '{subset}'. Available: {sorted(services)}")
    return services[subset].predict(req)
