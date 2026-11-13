"""Small prediction API backed by the promoted MLflow model."""

from functools import lru_cache
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from .config import ALIAS, model_uri, registry_uri, tracking_uri
from .data import FEATURES, NUMERIC


class Claim(BaseModel):
    initial_estimate: float = Field(ge=0)
    deductible: float = Field(ge=0)
    policy_tenure_months: int = Field(ge=0)
    prior_claims: int = Field(ge=0)
    reported_delay_days: int = Field(ge=0)
    incident_type: Literal["water", "theft", "fire", "wind"]
    region: Literal["northeast", "south", "midwest", "west"]


class Prediction(BaseModel):
    probability_high_cost: float
    predicted_high_cost: bool
    model_alias: str


@lru_cache(maxsize=1)
def get_model():
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_registry_uri(registry_uri() or tracking_uri())
    return mlflow.sklearn.load_model(model_uri())


app = FastAPI(title="Claims MLE 101", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(claim: Claim, model=Depends(get_model)) -> Prediction:
    frame = pd.DataFrame([claim.model_dump()])[FEATURES]
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    probability = float(model.predict_proba(frame)[0, 1])
    return Prediction(
        probability_high_cost=probability,
        predicted_high_cost=probability >= 0.5,
        model_alias=ALIAS,
    )

