from fastapi.testclient import TestClient
import numpy as np

from mle101.service import app, get_model


class ExampleModel:
    def predict_proba(self, frame):
        assert frame.iloc[0]["incident_type"] == "fire"
        return np.array([[0.2, 0.8]])


def test_predict_validates_and_returns_model_result():
    app.dependency_overrides[get_model] = lambda: ExampleModel()
    client = TestClient(app)
    claim = {
        "initial_estimate": 9000,
        "deductible": 1000,
        "policy_tenure_months": 12,
        "prior_claims": 1,
        "reported_delay_days": 4,
        "incident_type": "fire",
        "region": "south",
    }
    try:
        response = client.post("/predict", json=claim)
        assert response.status_code == 200
        assert response.json()["probability_high_cost"] == 0.8
        assert response.json()["predicted_high_cost"] is True
        claim["initial_estimate"] = -1
        assert client.post("/predict", json=claim).status_code == 422
    finally:
        app.dependency_overrides.clear()

