import pandas as pd
import pytest
from conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from turbofan_rul.api import app

    log = tmp_path_factory.mktemp("logs") / "pred.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("RUL_PREDICTION_LOG", str(log))
    with TestClient(app) as c:
        c.log_path = log
        yield c
    mp.undo()


def payload(df: pd.DataFrame) -> dict:
    return {"cycles": [{"cycle": int(r.cycle), "settings": [r.setting_1, r.setting_2, r.setting_3],
                        "sensors": [getattr(r, f"s{i}") for i in range(1, 22)]} for r in df.itertuples()]}


def test_health_lists_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "FD001" in body["models"]


def test_models_exposes_metrics(client):
    assert "rmse" in client.get("/models").json()["FD001"]["test_metrics"]


def test_predict_tracks_degradation(client, fd001):
    engine = fd001[fd001["unit"] == 1]  # 192 cycles to failure
    early = client.post("/predict/FD001", json=payload(engine.iloc[:40])).json()
    late = client.post("/predict/FD001", json=payload(engine.iloc[:-5])).json()
    assert early["rul_cycles"] > 90
    assert late["rul_cycles"] < 30 and not late["at_cap"]
    assert late["cycles_seen"] == len(engine) - 5


def test_predict_logs_latest_cycle(client, fd001):
    engine = fd001[fd001["unit"] == 2].iloc[:60].sample(frac=1, random_state=0)  # unordered
    client.post("/predict/FD001", json=payload(engine))
    last = pd.read_json(client.log_path, lines=True).iloc[-1]
    assert last["cycle"] == 60 and {"subset", "s11", "rul_cycles"} <= set(last.index)


def test_predict_rejects_bad_requests(client, fd001):
    engine = fd001[fd001["unit"] == 1].iloc[:10]
    assert client.post("/predict/FD999", json=payload(engine)).status_code == 404
    assert client.post("/predict/FD001", json=payload(pd.concat([engine, engine.iloc[[0]]]))).status_code == 422
    bad = payload(engine)
    bad["cycles"][0]["sensors"] = bad["cycles"][0]["sensors"][:5]
    assert client.post("/predict/FD001", json=bad).status_code == 422
