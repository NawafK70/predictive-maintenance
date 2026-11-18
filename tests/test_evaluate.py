import numpy as np
import pandas as pd
import pytest

from turbofan_rul.evaluate import evaluate, last_cycle_positions, nasa_score


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [50]) == 0
    early, late = nasa_score([50], [40]), nasa_score([50], [60])  # 10 cycles either way
    assert late > early > 0


def test_evaluate_capped_and_near_failure_metrics():
    y_true = np.array([10, 40, 200])
    y_pred = np.array([12, 40, 125])  # third engine is healthy: model predicts the cap
    m = evaluate(y_true, y_pred, rul_cap=125)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 0 + 75**2) / 3), abs=0.01)
    assert m["rmse_vs_capped_truth"] == pytest.approx(np.sqrt(4 / 3), abs=0.01)
    assert m["rmse_last_50_cycles"] == pytest.approx(np.sqrt(2), abs=0.01)
    assert m["pct_engines_above_cap"] == pytest.approx(33.3, abs=0.1)
    assert m["n_engines"] == 3


def test_last_cycle_positions_ordered_by_unit():
    df = pd.DataFrame({"unit": [2, 2, 1, 1, 1], "cycle": [1, 2, 1, 2, 3]})
    assert list(last_cycle_positions(df)) == [4, 1]
