import numpy as np
import pandas as pd

from turbofan_rul.monitor import drift_report, psi


def test_drift_report_flags_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(100, 10, 2000), "b": rng.normal(5, 1, 2000)})
    same = ref.sample(500, random_state=1)
    shifted = same.assign(a=same["a"] + 15)
    assert (drift_report(ref, same, ["a", "b"])["status"] == "stable").all()
    status = drift_report(ref, shifted, ["a", "b"]).set_index("feature")["status"]
    assert status["a"] == "DRIFT" and status["b"] == "stable"


def test_discrete_feature_with_float_noise_is_stable():
    """Regression: a setting with few levels, round-tripped through CSV/JSON, must not look drifted."""
    rng = np.random.default_rng(0)
    levels = np.round(np.arange(-5, 6) * 1e-4, 4)
    ref = rng.choice(levels, 5000)
    cur = rng.choice(levels, 300) + rng.choice([-1e-12, 0, 1e-12], 300)  # representation noise
    assert psi(ref, cur) < 0.1


def test_discrete_feature_shift_detected():
    ref = np.repeat([0.0, 1.0], [900, 100])
    cur = np.repeat([0.0, 1.0], [100, 900])
    assert psi(ref, cur) > 0.25
