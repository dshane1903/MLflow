"""Compare a new run with the persistent champion on one evaluation cohort."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ALIAS, MODEL_NAME, registry_uri
from .data import FEATURES, NUMERIC, TARGET, validate_claims
from .training import passes_gate


def _evaluate(model, frame: pd.DataFrame) -> dict[str, float]:
    features = frame[FEATURES].copy()
    features[NUMERIC] = features[NUMERIC].astype(float)
    probability = model.predict_proba(features)[:, 1]
    return {
        "average_precision": float(average_precision_score(frame[TARGET], probability)),
        "roc_auc": float(roc_auc_score(frame[TARGET], probability)),
    }


def compare_and_promote(
    summary_path: Path,
    evaluation_frame: pd.DataFrame,
    evaluation_source: str,
    min_ap_gain: float = 0.01,
) -> dict:
    """Promote only after evaluating candidate and champion on the same rows.

    On the first run there is no champion, so the candidate must beat evaluation
    prevalence by 0.05. Later runs must exceed the current champion's average
    precision by ``min_ap_gain``. The existing training gate also applies.
    """
    if min_ap_gain < 0:
        raise ValueError("min_ap_gain must be nonnegative")
    validate_claims(evaluation_frame, require_target=True)
    if evaluation_frame[TARGET].nunique() != 2:
        raise ValueError("evaluation cohort must contain both target classes")
    summary = json.loads(summary_path.read_text())
    mlflow.set_tracking_uri(summary["tracking_uri"])
    mlflow.set_registry_uri(registry_uri() or summary["tracking_uri"])
    client = MlflowClient(tracking_uri=summary["tracking_uri"], registry_uri=registry_uri())

    candidate = summary["experiments"]["logistic"]
    candidate_model = mlflow.sklearn.load_model(candidate["model_uri"])
    candidate_metrics = _evaluate(candidate_model, evaluation_frame)
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        current = None
    champion_metrics = None
    if current is not None:
        champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{current.version}")
        champion_metrics = _evaluate(champion_model, evaluation_frame)

    prevalence = float(evaluation_frame[TARGET].mean())
    passed_training_gate = passes_gate(summary["experiments"])
    passed_eval_floor = (
        candidate_metrics["average_precision"] >= prevalence + 0.05
        and candidate_metrics["roc_auc"] >= 0.65
    )
    beat_champion = (
        current is None
        or candidate_metrics["average_precision"]
        > champion_metrics["average_precision"] + min_ap_gain
    )
    should_promote = passed_training_gate and passed_eval_floor and beat_champion

    result = {
        "model_name": MODEL_NAME,
        "alias": ALIAS,
        "candidate_run_id": candidate["run_id"],
        "evaluation_source": evaluation_source,
        "evaluation_rows": len(evaluation_frame),
        "evaluation_prevalence": prevalence,
        "candidate": candidate_metrics,
        "previous_champion_version": current.version if current else None,
        "champion": champion_metrics,
        "minimum_ap_gain": min_ap_gain,
        "passed_training_gate": passed_training_gate,
        "passed_evaluation_floor": passed_eval_floor,
        "beat_champion": beat_champion,
        "promoted": should_promote,
    }
    client.log_metric(candidate["run_id"], "promotion_eval_average_precision", candidate_metrics["average_precision"])
    client.log_metric(candidate["run_id"], "promotion_eval_roc_auc", candidate_metrics["roc_auc"])
    client.set_tag(candidate["run_id"], "promotion_evaluation_source", evaluation_source)
    if champion_metrics:
        client.log_metric(candidate["run_id"], "champion_eval_average_precision", champion_metrics["average_precision"])
    if should_promote:
        version = mlflow.register_model(candidate["model_uri"], MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        result["new_champion_version"] = version.version
    else:
        result["new_champion_version"] = current.version if current else None
    client.set_tag(candidate["run_id"], "promotion_decision", "promoted" if should_promote else "rejected")
    (summary_path.parent / "promotion_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
