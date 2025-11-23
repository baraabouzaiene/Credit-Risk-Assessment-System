from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    # ignore extra keys like "model_version"


def test_predict_missing_features():
    resp = client.post("/predict", json={"features": {}})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "probability" in data


def test_predict_valid_payload():
    payload = {
        "features": {
            "LIMIT_BAL": 20000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 30,
            "PAY_0": 0,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 1000,
            "BILL_AMT2": 1000,
            "BILL_AMT3": 1000,
            "BILL_AMT4": 1000,
            "BILL_AMT5": 1000,
            "BILL_AMT6": 1000,
            "PAY_AMT1": 0,
            "PAY_AMT2": 0,
            "PAY_AMT3": 0,
            "PAY_AMT4": 0,
            "PAY_AMT5": 0,
            "PAY_AMT6": 0,
        }
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "probability" in data
    assert 0 <= data["probability"] <= 1


def test_predict_with_unknown_feature():
    payload = {"features": {"UNKNOWN_FEATURE": 123}}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert "prediction" in data
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
