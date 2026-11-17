# Predictive Maintenance

This project has two pipelines that share one FastAPI service, one Docker image and one workflow:

1. **Failure risk (classification)** uses the UCI **AI4I 2020** dataset. It takes six sensor and process values from a milling machine and returns the probability of failure.
2. **Remaining useful life (regression)** uses the **NASA C-MAPSS** turbofan dataset. It takes an engine's sensor history and returns how many operating cycles it has left before failure. This is the harder problem: it involves time series, multiple operating conditions and multiple fault modes.

```
pdm/
  config.py      paths, column names, constants
  data.py        AI4I: download + schema-validate
  features.py    AI4I: physics-informed features (inside the model pipeline)
  train.py       AI4I: compare models, tune threshold, evaluate, save
  rul/data.py      C-MAPSS: download, load, RUL labels
  rul/features.py  C-MAPSS: per-regime normalisation + causal rolling features
  rul/train.py     C-MAPSS: engine-grouped CV, model comparison, NASA scoring
  api.py         FastAPI: /predict, /predict/batch, /rul/{subset}/predict, /health, /model
  monitor.py     PSI drift check of live inputs vs training data
tests/           unit + API tests
Dockerfile       serving image (bakes in models/)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pdm.data      # download + summarise dataset
python -m pdm.train     # train, evaluate, write models/
python -m pdm.rul.data              # download + summarise C-MAPSS
python -m pdm.rul.train --subset all    # ~5–10 min; or --subset FD001
pytest -q
uvicorn pdm.api:app --reload    # then open http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"type":"L","air_temperature_k":302.5,"process_temperature_k":310.9,"rotational_speed_rpm":1320,"torque_nm":68,"tool_wear_min":215}'
# {"failure_probability":0.9864,"predicted_failure":true,"risk_level":"high","threshold":0.3662,...}
```

## Results (hold-out test set, 2,000 rows, 68 failures)

| Model (5-fold CV on train) | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.459 | 0.923 |
| **Random forest (selected)** | **0.894** | 0.977 |
| HistGradientBoosting | 0.886 | 0.975 |
| XGBoost | 0.861 | 0.976 |

XGBoost doesn't beat the random forest here, even after tuning. A 60-setting randomized search over depth, learning rate, regularisation and `scale_pos_weight` reached a best CV PR-AUC of 0.871. With only 10k rows and 6 inputs, the random forest's averaging over many deep trees is hard to beat. Boosting usually pulls ahead on larger, noisier, higher-dimensional data, such as the Azure PdM telemetry. `train.py` re-runs the comparison every time, so the winner can change as the data does.

Test results at the tuned threshold: **precision 0.89, recall 0.85, PR-AUC 0.88.** The confusion matrix is TP 58, FN 10, FP 7, TN 1925.

Recall by failure mode:

| Failure mode | Recall |
|---|---|
| Heat dissipation | 100% |
| Power | 100% |
| Overstrain | 100% |
| Tool wear | 20% |

Tool-wear failures are hard to predict because the dataset triggers them at random somewhere between 200 and 240 minutes of wear. In practice, handle them with a wear-based replacement rule rather than with the model.

## Remaining useful life – NASA C-MAPSS

The fleet data comes from simulated turbofan engines, each run until it fails. Every row is one operating cycle with 3 settings and 21 sensors. The four sub-datasets get harder:

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | 1 (HPC) | 100 / 100 |
| FD002 | 6 | 1 (HPC) | 260 / 259 |
| FD003 | 1 | 2 (HPC, fan) | 100 / 100 |
| FD004 | 6 | 2 (HPC, fan) | 249 / 248 |

**What makes it harder than AI4I:**
- **Time series.** One reading says little. The signal is in the *trend*, so features are causal rolling means and rolling slopes per engine.
- **Operating conditions.** In FD002/FD004, sensor values jump between 6 flight regimes far more than they drift with wear. The pipeline clusters the settings with KMeans and standardises each sensor *within its regime*. Without this step the degradation signal is buried.
- **Leakage trap.** Rows from the same engine are highly correlated. Cross-validation is grouped by engine (`GroupKFold`); a random split would be badly optimistic.
- **Label design.** A healthy engine shows no degradation, so "312 cycles left" can't be told apart from "180 cycles left". The target is capped at 125 cycles, the standard piecewise-linear RUL.
- **Asymmetric cost.** The NASA PHM08 score penalises late predictions (failure arrives before predicted) more heavily than early ones.

### Results

XGBoost won cross-validation on all four subsets; the random forest and HistGradientBoosting were within about 0.5 RMSE, and Ridge was far behind. The test set is NASA's official one, scored at each engine's last observed cycle.

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

### API

`POST /rul/{subset}/predict` takes an engine's cycle history: `cycle`, 3 `settings` and 21 `sensors` per cycle. It returns the remaining cycles plus `at_cap`, which means no measurable degradation yet. `GET /rul/models` lists the loaded models and their test metrics. These models are optional: the API serves whichever ones exist under `models/rul/`.

## Workflow

1. **Frame the problem.** What decision does the prediction drive, and how far ahead? Weigh the cost of a missed failure against a false alarm. Here a miss costs more, so the threshold maximises **F2** (recall-weighted). Change this with `--beta`.
2. **Data.** Check the schema, missing values and class balance. Failures are about 3% of rows. Drop leaking columns: the TWF/HDF/PWF/OSF/RNF failure-mode flags describe the failure itself.
3. **Features.** Domain physics beats generic tricks. The engineered features are temperature delta, mechanical power (torque × ω) and strain (wear × torque). They run *inside* the sklearn pipeline, so training and serving can't drift apart.
4. **Split correctly.** Use a stratified hold-out test set. With time-series data (C-MAPSS, real telemetry), split **by machine or by time** instead. A random split leaks future data.
5. **Model selection.** Start with a simple baseline, then move to tree ensembles, and compare them by **PR-AUC**. Accuracy is useless at 3% positives: always predicting "OK" scores 97%.
6. **Threshold tuning.** Pick the threshold from out-of-fold CV predictions, never from the test set.
7. **Evaluate once on test.** Report recall per failure mode and permutation importance. These are what the maintenance team will ask about.
8. **Package.** The pipeline goes in `model.joblib`. `metadata.json` holds the version, threshold, metrics and library versions. `reference.csv` is the drift baseline.
9. **Serve.** FastAPI validates every input (ranges, allowed types) and returns a probability, a yes/no flag and a risk level. Each request is logged to JSONL.
10. **Monitor & retrain.** Run `python -m pdm.monitor` on a schedule. It exits with code 1 when PSI > 0.25 on any feature, which can trigger a retraining job. Once technicians confirm real failures, also track realised precision and recall.

## Deployment

```bash
python -m pdm.train            # the image bakes in models/
docker build -t pdm-api .
docker run -p 8000:8000 -v ${PWD}/logs:/app/logs pdm-api
```

Docker Desktop must be running. `requirements-serve.txt` pins scikit-learn to the version used for training; a mismatch can break `joblib.load`.

How far to take deployment, from simplest to most complete:

- **Simplest:** run the container on a VM, or on a managed container service such as Azure Container Apps, AWS App Runner or Google Cloud Run.
- **Batch scoring:** if the readings arrive as a nightly file or database table rather than live, a scheduled job that calls `pipeline.predict_proba` is simpler than an API.
- **Production MLOps:** add MLflow for experiment tracking and a model registry, so the container pulls the "Production" model instead of baking it in. Add CI that runs `pytest` and a minimum-PR-AUC gate before promoting a model, and Prometheus/Grafana for latency and prediction-rate dashboards.
- **Edge/IoT:** export to ONNX (`skl2onnx`) and run it on the gateway next to the PLC. Readings can come in over MQTT or OPC-UA.

## Dataset suggestions

| Dataset | Task | Size | Notes |
|---|---|---|---|
| **AI4I 2020** (UCI) – *used here* | Failure classification + failure mode | 10k rows | Clean, synthetic but realistic; best starting point |
| **NASA C-MAPSS** turbofan – *used here* | Remaining useful life (RUL) regression | 4 subsets, ~100–260 engines each | The standard RUL benchmark; requires windowing, sequence models (LSTM/1D-CNN) or rolling features |
| **Microsoft Azure PdM** (Kaggle) | Failure within next 24h | 100 machines, 1 year hourly telemetry + errors + maintenance logs | Closest to real plant data: joins telemetry, error logs and maintenance history |
| **CWRU Bearing** | Fault diagnosis from vibration | High-frequency signals | Signal processing (FFT, envelope spectra) + classification |
| **IMS / NASA Bearing**, **FEMTO/PRONOSTIA** | Bearing degradation / RUL | Run-to-failure vibration | Good for anomaly detection and health-index approaches |
| **MetroPT-3** (UCI) | Anomaly / failure detection | Train air compressor, 15M rows | Real-world, unlabeled-ish, good for unsupervised methods |
| **SECOM** (UCI) | Semiconductor process fail/pass | 1.5k rows, 590 sensors | High-dimensional, feature selection heavy |

Suggested learning path: AI4I (classification), then C-MAPSS (RUL regression), then Azure PdM (multi-source joins and a time-based train/test split).

## Next steps

- Multi-class or multi-label model to predict *which* failure mode will happen.
- Probability calibration (`CalibratedClassifierCV`) if the probabilities feed cost calculations.
- SHAP explanations per prediction, returned from the API.
- C-MAPSS: a 1D-CNN or LSTM over 30-cycle windows. This is the usual next step to get below about 13 RMSE.
- C-MAPSS: prediction intervals (quantile regression or conformal), so maintenance can plan around "RUL 40 ± 12" rather than a single number.
- C-MAPSS: train one model on FD001–FD004 combined, with a fault-mode or regime indicator.
