#this checks that  the processed training data has the same number of features as during training


from pathlib import Path
import pandas as pd

X_TRAIN_PATH = Path("data/processed/X_train.csv")

def test_expected_columns_count():
    assert X_TRAIN_PATH.exists(), "X_train.csv not found in data/processed/"
    df = pd.read_csv(X_TRAIN_PATH)
    n_features = df.shape[1]
    assert n_features == 102, f"Expected 102 features, got {n_features}"
