import numpy as np
import pandas as pd
import pytest
from conftest import requires_data

from turbofan_rul.features import RULFeaturizer

pytestmark = requires_data


@pytest.fixture(scope="module")
def featurizer(fd001):
    return RULFeaturizer(n_conditions=1).fit(fd001[fd001["unit"] <= 20])


def test_features_are_causal(fd001, featurizer):
    """Appending future cycles must not change the features of earlier cycles."""
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
    assert "s1" not in featurizer.sensors_  # constant at sea level
    assert "s11" in featurizer.sensors_


def test_missing_columns_rejected(featurizer, fd001):
    with pytest.raises(ValueError, match="missing columns"):
        featurizer.transform(fd001.drop(columns=["s11"]))


def test_finds_six_operating_conditions():
    from turbofan_rul.data import load_train
    df = load_train("FD002")
    f = RULFeaturizer(n_conditions=6).fit(df)
    counts = np.bincount(f.conditions(df))
    assert len(counts) == 6 and counts.min() > 0.1 * len(df)
    # Raw sensors jump between regimes by far more than the within-regime noise;
    # after per-condition normalisation those jumps are gone.
    engine = df[df["unit"] == 1]
    assert engine["s11"].max() - engine["s11"].min() > 10 * f.std_["s11"].max()
    assert f.normalize(engine).abs().max().max() < 6
