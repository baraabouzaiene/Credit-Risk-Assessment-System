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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "lightgbm_best",
    }

class FeaturesDict(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "lightgbm_best"

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeaturesDict):
    try:
        # Incoming dict from JSON body
        raw_feats = payload.features or {}

        # 1) Start with a default row: all model features = 0.0
        row = {col: 0.0 for col in EXPECTED_COLUMNS}

        # 2) Overwrite with any known feature from the payload
        #    Unknown features are IGNORED (no 422)
        for k, v in raw_feats.items():
            if k in row:
                row[k] = v
            # else: ignore unknown keys silently

        # 3) Build DataFrame in the exact same column order as training
        df = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

        # 4) Predict probability of default (class 1)
        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)

        return PredictionResponse(
            prediction=pred,
            probability=proba,
            model_version="lightgbm_best",
        )

    except HTTPException:
        # re-raise if we throw it intentionally somewhere else
        raise
    except Exception as e:
        # any unexpected error → 500
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
