import numpy as np
import pandas as pd
import pytest

from pdm import config
from pdm.features import add_features
from pdm.monitor import drift_report

HIGH_RISK = {  # low speed + high torque + worn tool -> overstrain / power failure territory
    "type": "L", "air_temperature_k": 302.5, "process_temperature_k": 310.9,
    "rotational_speed_rpm": 1320, "torque_nm": 68.0, "tool_wear_min": 215,
}
LOW_RISK = {
    "type": "H", "air_temperature_k": 298.0, "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1550, "torque_nm": 40.0, "tool_wear_min": 10,
}


def test_add_features():
    out = add_features(pd.DataFrame([LOW_RISK]))
    assert out["temp_diff_k"].iloc[0] == pytest.approx(10.6)
    assert out["power_w"].iloc[0] == pytest.approx(40 * 1550 * 2 * np.pi / 60)
    assert out["strain_min_nm"].iloc[0] == pytest.approx(400)


def test_drift_report_flags_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], 2000),
        **{c: rng.normal(100, 10, 2000) for c in config.NUMERIC},
    })
    same = ref.sample(500, random_state=1)
    shifted = same.assign(torque_nm=same["torque_nm"] + 15)
    assert (drift_report(ref, same)["status"] == "stable").all()
    status = drift_report(ref, shifted).set_index("feature")["status"]
    assert status["torque_nm"] == "DRIFT"
    assert status["tool_wear_min"] == "stable"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not (config.MODEL_DIR / config.MODEL_FILE).exists():
        pytest.skip("no trained model; run python -m pdm.train")
    from fastapi.testclient import TestClient
    log = tmp_path_factory.mktemp("logs") / "pred.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("PDM_PREDICTION_LOG", str(log))
    from pdm.api import app
    with TestClient(app) as c:
        c.log_path = log
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_risk_ordering(client):
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["failure_probability"] > low["failure_probability"]
    assert high["predicted_failure"] is True and high["risk_level"] == "high"
    assert low["predicted_failure"] is False and low["risk_level"] == "low"


def test_batch_and_logging(client):
    r = client.post("/predict/batch", json={"readings": [HIGH_RISK, LOW_RISK, LOW_RISK]})
    assert r.status_code == 200 and len(r.json()) == 3
    logged = pd.read_json(client.log_path, lines=True)
    assert set(config.FEATURES) <= set(logged.columns)


@pytest.mark.parametrize("bad", [{**LOW_RISK, "type": "X"}, {**LOW_RISK, "torque_nm": -5}, {"type": "L"}])
def test_rejects_invalid_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 422
