from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_prediction_request_validation():
    response = client.post("/api/predict", json={"MedInc":-1})
    assert response.status_code == 422

def test_bad_tuning_model():
    response = client.post("/api/tune", json={"model":"Linear Regression"})
    assert response.status_code == 400
