"""Experiment tracking and a deliberate model-promotion gate."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ALIAS, MODEL_NAME, ROOT, experiment_name, registry_uri, tracking_uri
from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET, validate_claims
from .monitoring import make_reference


def make_pipeline(estimator: object) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categories = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC), ("categorical", categories, CATEGORICAL)]
                ),
            ),
            ("model", estimator),
        ]
    )


def train(data_path: Path, output_dir: Path | None = None) -> dict:
    frame = pd.read_csv(data_path)
    return train_frame(frame, str(data_path.resolve()), output_dir)


def train_frame(frame: pd.DataFrame, data_source: str, output_dir: Path | None = None) -> dict:
    """Train on a bounded pandas frame from CSV or a curated Delta table."""
    validate_claims(frame, require_target=True)
    output_dir = output_dir or ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratification keeps class proportions comparable in this small lesson.
    train_frame, test_frame = train_test_split(
        frame, test_size=0.25, stratify=frame[TARGET], random_state=42
    )
    x_train, y_train = train_frame[FEATURES].copy(), train_frame[TARGET]
    x_test, y_test = test_frame[FEATURES].copy(), test_frame[TARGET]
    for column in NUMERIC:
        x_train[column] = x_train[column].astype(float)
        x_test[column] = x_test[column].astype(float)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_registry_uri(registry_uri() or tracking_uri())
    mlflow.set_experiment(experiment_name())
    results = {}

    candidates = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
    }
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(x_train, y_train)
            train_probability = pipeline.predict_proba(x_train)[:, 1]
            test_probability = pipeline.predict_proba(x_test)[:, 1]
            metrics = {
                "train_roc_auc": float(roc_auc_score(y_train, train_probability)),
                "test_roc_auc": float(roc_auc_score(y_test, test_probability)),
                "test_average_precision": float(
                    average_precision_score(y_test, test_probability)
                ),
                "test_prevalence": float(y_test.mean()),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {"algorithm": name, "rows_train": len(x_train), "rows_test": len(x_test)}
            )
            mlflow.set_tags({"dataset": "synthetic", "data_source": data_source, "target": TARGET, "seed": "42"})
            signature = infer_signature(x_test.head(5), test_probability[:5])
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=x_test.head(5),
                signature=signature,
                skops_trusted_types=["numpy.dtype"],
            )
            results[name] = {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                **metrics,
            }

    reference = make_reference(x_train)
    (output_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    summary = {
        "tracking_uri": tracking_uri(),
        "data_source": data_source,
        "reference_path": str((output_dir / "reference.json").resolve()),
        "experiments": results,
        "recommended": "logistic" if passes_gate(results) else None,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def passes_gate(results: dict) -> bool:
    """An explicit teaching gate, not an industry-standard threshold."""
    base = results["baseline"]
    candidate = results["logistic"]
    return (
        candidate["test_average_precision"] >= base["test_average_precision"] + 0.05
        and candidate["test_roc_auc"] >= 0.65
        and candidate["train_roc_auc"] - candidate["test_roc_auc"] <= 0.15
    )


def promote(summary_path: Path, name: str = "logistic") -> dict:
    summary = json.loads(summary_path.read_text())
    if name != "logistic" or not passes_gate(summary["experiments"]):
        raise ValueError("promotion gate failed; inspect the MLflow runs before promoting")
    mlflow.set_tracking_uri(summary["tracking_uri"])
    mlflow.set_registry_uri(registry_uri() or summary["tracking_uri"])
    client = MlflowClient()
    info = summary["experiments"][name]
    version = mlflow.register_model(model_uri=info["model_uri"], name=MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    result = {"name": MODEL_NAME, "version": version.version, "alias": ALIAS}
    (summary_path.parent / "promotion.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

