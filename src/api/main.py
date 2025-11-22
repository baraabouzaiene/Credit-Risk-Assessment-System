from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Credit Risk Assessment API",
    version="1.0.0",
    description="Predict default risk using a trained LightGBM model",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_best.pkl"
X_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.csv"

model = joblib.load(MODEL_PATH)

# Columns the model was trained on (all 102)
EXPECTED_COLUMNS = list(pd.read_csv(X_TRAIN_PATH, nrows=1).columns)


class FeaturesDict(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "lightgbm_best"

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeaturesDict):
    try:
        raw_feats = payload.features  # dict from JSON

        # 1) Start with a default row: all features = 0.0
        row = {col: 0.0 for col in EXPECTED_COLUMNS}

        # 2) Overwrite with values that match known columns
        for k, v in raw_feats.items():
            if k in row:
                row[k] = v
            # else: silently ignore unknown keys for now

        # 3) Build DataFrame in the exact training column order
        df = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

        # 4) Predict
        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)

        return PredictionResponse(
            prediction=pred,
            probability=proba,
            model_version="lightgbm_best",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
