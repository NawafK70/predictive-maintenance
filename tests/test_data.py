from conftest import requires_data

from turbofan_rul import settings
from turbofan_rul.data import COLUMNS, load_test


@requires_data
def test_train_rul_label(fd001):
    assert (fd001.groupby("unit").tail(1)["rul"] == 0).all()
    assert (fd001.groupby("unit")["rul"].diff().dropna() == -1).all()  # counts down one per cycle


@requires_data
def test_test_set_shapes():
    for subset in settings.SUBSETS:
        df, rul = load_test(subset)
        assert list(df.columns) == COLUMNS
        assert len(rul) == df["unit"].nunique()
        assert (rul > 0).all()
