import numpy as np
import pandas as pd
import pytest

from pdm import config

pytestmark = pytest.mark.skipif(
    not (config.CMAPSS_DIR / "train_FD001.txt").exists(), reason="C-MAPSS not downloaded (python -m pdm.rul.data)"
)


@pytest.fixture(scope="module")
def fd001():
    from pdm.rul.data import load_train
    return load_train("FD001")


@pytest.fixture(scope="module")
def featurizer(fd001):
    from pdm.rul.features import RULFeaturizer
    return RULFeaturizer(n_conditions=1).fit(fd001[fd001["unit"] <= 20])


def test_train_rul_label(fd001):
    last = fd001.groupby("unit").tail(1)
    assert (last["rul"] == 0).all()
    assert fd001["rul"].min() == 0


def test_features_are_causal(fd001, featurizer):
    """Appending future cycles must not change features of earlier cycles."""
    engine = fd001[fd001["unit"] == 1]
    full = featurizer.transform(engine)
    partial = featurizer.transform(engine.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50].reset_index(drop=True), partial.reset_index(drop=True))


def test_transform_preserves_row_order(fd001, featurizer):
    sample = fd001[fd001["unit"].isin([1, 2])]
    shuffled = sample.sample(frac=1, random_state=0)
    a = featurizer.transform(sample).set_axis(sample.index)
    b = featurizer.transform(shuffled).set_axis(shuffled.index)
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_constant_sensors_dropped(featurizer):
    assert "s1" not in featurizer.sensors_  # s1 is constant at sea level
    assert "s11" in featurizer.sensors_


def test_finds_six_operating_conditions():
    from pdm.rul.data import load_train
    from pdm.rul.features import RULFeaturizer
    df = load_train("FD002")
    f = RULFeaturizer(n_conditions=6).fit(df)
    counts = np.bincount(f._conditions(df))
    assert len(counts) == 6 and counts.min() > 0.1 * len(df)
    # After per-condition normalisation the regime jumps disappear: z-scores stay small.
    z = f.transform(df[df["unit"] == 1])
    assert z.filter(like="_mean5").abs().max().max() < 6


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not (config.RUL_MODEL_DIR / "FD001" / config.MODEL_FILE).exists():
        pytest.skip("no RUL model; run python -m pdm.rul.train")
    from fastapi.testclient import TestClient
    mp = pytest.MonkeyPatch()
    mp.setenv("PDM_PREDICTION_LOG", "")
    from pdm.api import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def _payload(df):
    return {"cycles": [{"cycle": int(r.cycle), "settings": [r.setting_1, r.setting_2, r.setting_3],
                        "sensors": [getattr(r, f"s{i}") for i in range(1, 22)]} for r in df.itertuples()]}


def test_rul_endpoint_tracks_degradation(client, fd001):
    engine = fd001[fd001["unit"] == 1]  # 192 cycles to failure
    early = client.post("/rul/FD001/predict", json=_payload(engine.iloc[:40])).json()
    late = client.post("/rul/FD001/predict", json=_payload(engine.iloc[:-5])).json()
    assert early["rul_cycles"] > 90
    assert late["rul_cycles"] < 30 and not late["at_cap"]
    assert late["cycles_seen"] == len(engine) - 5


def test_rul_endpoint_errors(client, fd001):
    engine = fd001[fd001["unit"] == 1].iloc[:10]
    assert client.post("/rul/FD999/predict", json=_payload(engine)).status_code == 404
    dup = _payload(pd.concat([engine, engine.iloc[[0]]]))
    assert client.post("/rul/FD001/predict", json=dup).status_code == 422
    bad = _payload(engine)
    bad["cycles"][0]["sensors"] = bad["cycles"][0]["sensors"][:5]
    assert client.post("/rul/FD001/predict", json=bad).status_code == 422
