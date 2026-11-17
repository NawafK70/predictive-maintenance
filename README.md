# Predictive Maintenance – Turbofan Remaining Useful Life

This project predicts how many operating cycles a turbofan engine has left before it fails, known as its **remaining useful life (RUL)**. It is trained on the **NASA C-MAPSS** run-to-failure dataset and served through a FastAPI REST API. The API ships with a Docker image and a data-drift monitor.

```
pdm/
  config.py    paths and constants
  data.py      download C-MAPSS, load, RUL labels
  features.py  per-regime normalisation + causal rolling features (inside the model pipeline)
  train.py     engine-grouped CV, model comparison, NASA scoring, save
  api.py       FastAPI: /predict/{subset}, /models, /health
  monitor.py   PSI drift check of live readings vs training data
demo/app.py    Streamlit demo: replay a test engine, watch the prediction evolve
tests/         unit + API tests
Dockerfile     serving image (bakes in models/)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pdm.data                  # download + summarise C-MAPSS
python -m pdm.train --subset all    # ~5–10 min; or --subset FD001
pytest -q
streamlit run demo/app.py           # interactive demo, http://localhost:8501
uvicorn pdm.api:app --reload        # REST API, http://localhost:8000/docs
```

## Demo

`demo/app.py` is a Streamlit app that replays engines from NASA's held-out test set. The model never saw these engines during training.

- **Pick an engine:** choose a subset and an engine. Engines with the longest histories are listed first.
- **Replay it:** drag the slider through its life. At each cycle you see:
  - the predicted and actual remaining life
  - a maintenance status: healthy, plan maintenance, or critical
  - the full predicted-vs-actual trajectory, with a hover tooltip
  - the four most informative sensors, as the model sees them: normalised per operating condition and smoothed
- **Behind the scenes:** the app loads the trained pipelines from `models/` directly, so it doesn't need the API running.

## The dataset

C-MAPSS comes from simulated turbofan engines, each run until it fails. Every row is one engine at one operating cycle: 3 operational settings and 21 sensor readings. Training engines run all the way to failure. Test engines stop some time before failure, and NASA provides their true RUL. The four sub-datasets get harder:

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | 1 (HPC) | 100 / 100 |
| FD002 | 6 | 1 (HPC) | 260 / 259 |
| FD003 | 1 | 2 (HPC, fan) | 100 / 100 |
| FD004 | 6 | 2 (HPC, fan) | 249 / 248 |

**What makes it hard:**
- **Time series.** One reading says little. The signal is in the *trend*, so features are causal rolling means and rolling slopes per engine. A causal feature uses only past cycles, which a test checks.
- **Operating conditions.** In FD002/FD004, sensor values jump between 6 flight regimes far more than they drift with wear. The pipeline clusters the settings with KMeans and standardises each sensor *within its regime*. Without this step the degradation signal is buried.
- **Leakage trap.** Rows from the same engine are highly correlated. Cross-validation is grouped by engine (`GroupKFold`); a random split would be badly optimistic.
- **Label design.** A healthy engine shows no degradation, so "312 cycles left" can't be told apart from "180 cycles left". The target is capped at 125 cycles, the standard piecewise-linear RUL.
- **Asymmetric cost.** The NASA PHM08 score penalises late predictions (failure arrives before predicted) more heavily than early ones.

## Results

Four models were compared by engine-grouped 5-fold CV: Ridge, random forest, HistGradientBoosting and XGBoost. XGBoost won on all four subsets; the random forest and HistGradientBoosting were within about 0.5 RMSE, and Ridge was far behind. The test set is NASA's official one, scored at each engine's last observed cycle.

| Subset | CV RMSE | Test RMSE | vs capped truth | Last 50 cycles | NASA score | Within ±15 cycles |
|---|---|---|---|---|---|---|
| FD001 | 13.9 | 15.5 | 14.2 | 6.7 | 375 | 71% |
| FD002 | 15.1 | 26.5 | 14.4 | 7.9 | 8,732 | 59% |
| FD003 | 11.9 | 15.5 | 13.6 | 4.8 | 387 | 73% |
| FD004 | 14.5 | 26.8 | 15.0 | 9.7 | 4,562 | 55% |

**How to read this:**
- **FD002/FD004 test RMSE is not a bug.** 22–27% of their test engines are still healthy, with up to about 190 cycles left. The capped model can't predict above 125. Against capped truth, all four subsets score 14–15, consistent with CV.
- **What matters operationally is the "last 50 cycles" column.** When an engine is within 50 cycles of failure, predictions are off by 5–10 cycles. That is the window where maintenance gets scheduled.
- **For context:** published deep-learning results (CNN/LSTM/Transformer) are around 11–13 RMSE on FD001 and 13–20 on FD004. Papers differ on whether test truth is capped, so compare carefully.

## API

`POST /predict/{subset}` takes an engine's cycle history: `cycle`, 3 `settings` and 21 `sensors` per cycle, oldest first or in any order. It returns:

```json
{"rul_cycles": 23.4, "at_cap": false, "rul_cap": 125, "cycles_seen": 187, "subset": "FD001", "model_version": "..."}
```

`at_cap: true` means the engine shows no measurable degradation yet: its RUL is at least 125 cycles, possibly much more. `GET /models` lists the loaded models with their test metrics. `GET /health` is the liveness check. The API loads every subset found under `models/` and refuses to start if there are none.

## Workflow

1. **Frame the problem.** What decision does the prediction drive, and how far ahead? Here it is scheduling maintenance before failure. Predicting failure *too late* is the costly error, which is why the NASA score is asymmetric.
2. **Data.** Check the schema, run-to-failure lengths and operating regimes (`python -m pdm.data`). Drop constant sensors.
3. **Label.** RUL = last cycle − current cycle, capped at 125.
4. **Features.** Normalise each sensor per regime, then add per-engine rolling means and slopes plus engine age. Everything runs *inside* the sklearn pipeline, so training and serving compute features identically.
5. **Split correctly.** Group by engine. With real fleet data, also split by time, training on older engines and testing on newer ones.
6. **Model selection.** Start with a linear baseline, then move to tree ensembles, and compare them by CV RMSE.
7. **Evaluate once on test.** Score at the last observed cycle per engine. Report RMSE, RMSE near failure, the NASA score and the share of late predictions.
8. **Package.** The pipeline goes in `model.joblib`. `metadata.json` holds the version, metrics, sensors used and library versions. `reference.csv` is the drift baseline.
9. **Serve.** FastAPI validates every request (3 settings, 21 sensors, no duplicate cycles) and returns RUL. The latest reading of each request is logged to JSONL.
10. **Monitor & retrain.** Run `python -m pdm.monitor --subset FD001` on a schedule. It exits with code 1 when PSI > 0.25 on any setting or sensor, which can trigger retraining. Once engines actually fail, also compare predicted against actual RUL. Degradation itself shifts sensors, so read drift alongside the predicted-RUL distribution.

## Deployment

```bash
python -m pdm.train --subset all    # the image bakes in models/
docker build -t pdm-api .
docker run -p 8000:8000 -v ${PWD}/logs:/app/logs pdm-api
```

Docker Desktop must be running. `requirements-serve.txt` pins scikit-learn and XGBoost to the versions used for training; a mismatch can break `joblib.load`.

How far to take deployment, from simplest to most complete:

- **Simplest:** run the container on a VM, or on a managed container service such as Azure Container Apps, AWS App Runner or Google Cloud Run.
- **Batch scoring:** if engine data arrives as files after each flight, a scheduled job that scores every engine's history is simpler than an API.
- **Production MLOps:** add MLflow for experiment tracking and a model registry, so the container pulls the "Production" model instead of baking it in. Add CI that runs `pytest` and a maximum-RMSE gate before promoting a model, and Prometheus/Grafana for latency and prediction dashboards.
- **Edge:** export to ONNX and run it on the on-board or ground-station gateway.

## Other datasets worth trying

| Dataset | Task | Notes |
|---|---|---|
| **Microsoft Azure PdM** (Kaggle) | Failure within next 24h | 100 machines, a year of hourly telemetry + error, maintenance and failure logs; closest to real plant data |
| **MetroPT-3** (UCI) | Anomaly / failure detection | Real train air-compressor data, 15M rows, few labels; good for unsupervised methods |
| **IMS / NASA Bearing**, **FEMTO/PRONOSTIA** | Bearing degradation / RUL | Run-to-failure vibration; health-index approaches |
| **CWRU Bearing** | Fault diagnosis from vibration | Signal processing (FFT, envelope spectra) + classification |
| **N-CMAPSS** (NASA) | RUL under real flight profiles | The newer, much larger successor to C-MAPSS |

## Next steps

- A 1D-CNN or LSTM over 30-cycle windows. This is the usual next step to get below about 13 RMSE.
- Prediction intervals (quantile regression or conformal), so maintenance can plan around "RUL 40 ± 12" rather than a single number.
- One model trained on FD001–FD004 combined, with a regime indicator.
- SHAP explanations of which sensors drive each prediction.
