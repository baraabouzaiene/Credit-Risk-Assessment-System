# this checks that the lightgbm_best.pkl exists and loadd

from pathlib import Path
import joblib

def test_model_exists_and_loads():
    path = Path("models/lightgbm_best.pkl")
    assert path.exists(), "Model file not found at models/lightgbm_best.pkl"

    model = joblib.load(path)
    assert model is not None