"""Paths and model settings shared by the command line and API."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "mle101-claims-high-cost"
ALIAS = "champion"


def tracking_uri() -> str:
    return os.getenv("MLE101_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

