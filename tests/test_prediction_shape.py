# checks that model outputs the right shape for predict_proba



import joblib
import pandas as pd

def test_prediction_probability_shape():
    model = joblib.load("models/lightgbm_best.pkl")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_sample = X_train.head(5)

    proba = model.predict_proba(X_sample)

    assert proba.shape == (5, 2), f"Expected (5, 2), got {proba.shape}"
