import numpy as np
from conftest import requires_model

from turbofan_rul.inference import engine_trajectory

pytestmark = requires_model


def test_predictions_within_cap(fd001_model, fd001):
    pred = fd001_model.predict(fd001[fd001["unit"] <= 5])
    assert pred.min() >= 0 and pred.max() <= fd001_model.rul_cap


def test_predict_latest_ignores_row_order(fd001_model, fd001):
    engine = fd001[fd001["unit"] == 3]
    shuffled = engine.sample(frac=1, random_state=0)
    assert fd001_model.predict_latest(engine) == fd001_model.predict_latest(shuffled)


def test_trajectory_matches_truncated_histories(fd001_model, fd001):
    """Prediction at cycle t from the full trajectory equals predicting with cycles 1..t only."""
    engine = fd001[fd001["unit"] == 7]
    traj = engine_trajectory(fd001_model, engine)
    for t in (30, 100):
        assert np.isclose(traj.loc[traj["cycle"] == t, "predicted"].item(),
                          round(fd001_model.predict_latest(engine[engine["cycle"] <= t]), 1))


def test_metadata_is_traceable(fd001_model):
    meta = fd001_model.metadata
    assert {"config", "model_comparison", "test_metrics", "versions", "sensors_used"} <= set(meta)
    assert meta["model_name"] in meta["config"]["models"]
