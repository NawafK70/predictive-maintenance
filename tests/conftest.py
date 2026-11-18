import pandas as pd
import pytest

from turbofan_rul import settings

HAS_DATA = (settings.DATA_DIR / "train_FD001.txt").exists()
HAS_MODEL = (settings.MODEL_DIR / "FD001" / settings.MODEL_FILE).exists()

requires_data = pytest.mark.skipif(not HAS_DATA, reason="C-MAPSS not downloaded (run rul-data)")
requires_model = pytest.mark.skipif(not (HAS_DATA and HAS_MODEL), reason="no FD001 model (run rul-train)")


@pytest.fixture(scope="session")
def fd001() -> pd.DataFrame:
    from turbofan_rul.data import load_train
    return load_train("FD001")


@pytest.fixture(scope="session")
def fd001_model():
    from turbofan_rul.inference import RULPredictor
    return RULPredictor.load(settings.MODEL_DIR / "FD001")
