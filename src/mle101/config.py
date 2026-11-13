"""Paths and model settings shared by the command line and API."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = os.getenv("MLE101_MODEL_NAME", "mle101-claims-high-cost")
ALIAS = "champion"


def tracking_uri() -> str:
    return os.getenv("MLE101_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def registry_uri() -> str | None:
    return os.getenv("MLE101_REGISTRY_URI")


def experiment_name() -> str:
    return os.getenv("MLE101_EXPERIMENT_NAME", "mle101-claims")


def model_uri() -> str:
    return os.getenv("MLE101_MODEL_URI", f"models:/{MODEL_NAME}@{ALIAS}")

