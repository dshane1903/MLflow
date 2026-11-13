"""Input drift checks for a newly scored batch (no labels required)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC, validate_claims


def make_reference(frame: pd.DataFrame) -> dict:
    reference = {"numeric": {}, "categorical": {}, "rows": len(frame)}
    for column in NUMERIC:
        # Unique interior quantiles prevent empty or duplicate bins.
        interior = np.unique(np.quantile(frame[column], np.linspace(0.1, 0.9, 9)))
        edges = np.r_[-np.inf, interior, np.inf]
        counts = np.histogram(frame[column], bins=edges)[0]
        reference["numeric"][column] = {
            "edges": [float(x) for x in interior],
            "proportions": (counts / counts.sum()).tolist(),
        }
    for column in CATEGORICAL:
        reference["categorical"][column] = frame[column].value_counts(normalize=True).to_dict()
    return reference


def psi(expected: list[float], observed: list[float]) -> float:
    """Population stability index with smoothing for empty bins."""
    left = np.maximum(np.asarray(expected, dtype=float), 1e-6)
    right = np.maximum(np.asarray(observed, dtype=float), 1e-6)
    return float(np.sum((right - left) * np.log(right / left)))


def monitor(reference_path: Path, scored_path: Path, report_path: Path) -> dict:
    reference = json.loads(reference_path.read_text())
    frame = pd.read_csv(scored_path)
    validate_claims(frame)
    scores = {}
    for column, ref in reference["numeric"].items():
        edges = np.r_[-np.inf, ref["edges"], np.inf]
        observed = np.histogram(frame[column], bins=edges)[0] / len(frame)
        scores[column] = round(psi(ref["proportions"], observed.tolist()), 4)
    for column, ref in reference["categorical"].items():
        labels = sorted(set(ref) | set(frame[column]))
        expected = [ref.get(label, 0) for label in labels]
        observed = frame[column].value_counts(normalize=True)
        scores[column] = round(psi(expected, [observed.get(label, 0) for label in labels]), 4)
    # 0.2 is an exercise threshold. It is not a universal production alert level.
    flagged = sorted(column for column, value in scores.items() if value > 0.2)
    report = {
        "reference_rows": reference["rows"],
        "scored_rows": len(frame),
        "psi_by_feature": scores,
        "exercise_threshold": 0.2,
        "flagged_features": flagged,
        "action": "investigate incoming data and model performance" if flagged else "continue monitoring",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

