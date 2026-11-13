"""Commands for the complete local learning loop."""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from .config import ROOT, model_uri, registry_uri, tracking_uri
from .data import FEATURES, NUMERIC, make_claims, validate_claims
from .monitoring import monitor
from .promotion import compare_and_promote
from .training import promote, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mle101", description="Claims MLOps learning loop")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate synthetic claim records")
    generate.add_argument("--rows", type=int, default=2500)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--shifted", action="store_true")
    generate.add_argument("--output", type=Path, default=ROOT / "data" / "claims.csv")

    train_cmd = commands.add_parser("train", help="track baseline and candidate experiments")
    train_input = train_cmd.add_mutually_exclusive_group()
    train_input.add_argument("--input", type=Path)
    train_input.add_argument("--delta-table", type=str)
    train_cmd.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")

    spark_build = commands.add_parser("spark-build", help="validate CSV with PySpark and write Delta tables")
    spark_build.add_argument("--input", type=str, required=True)
    spark_build.add_argument("--bronze-table", type=str, required=True)
    spark_build.add_argument("--silver-table", type=str, required=True)

    promote_cmd = commands.add_parser("promote", help="register a passing model as champion")
    promote_cmd.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts" / "training_summary.json"
    )

    auto = commands.add_parser("auto-promote", help="compare candidate with champion on one cohort")
    auto.add_argument("--summary", type=Path, default=ROOT / "artifacts" / "training_summary.json")
    evaluation = auto.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-input", type=Path)
    evaluation.add_argument("--eval-delta-table", type=str)
    auto.add_argument("--min-ap-gain", type=float, default=0.01)

    export = commands.add_parser("export-model", help="save the champion for local Docker serving")
    export.add_argument("--output", type=Path, default=ROOT / "artifacts" / "container-model")

    score = commands.add_parser("score", help="score a batch with the champion model")
    score.add_argument("--input", type=Path, default=ROOT / "data" / "incoming.csv")
    score.add_argument("--output", type=Path, default=ROOT / "artifacts" / "scored.csv")

    check = commands.add_parser("monitor", help="compare the new batch to training data")
    check.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts" / "reference.json"
    )
    check.add_argument("--scored", type=Path, default=ROOT / "artifacts" / "scored.csv")
    check.add_argument("--output", type=Path, default=ROOT / "artifacts" / "monitoring.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        frame = make_claims(n=args.rows, seed=args.seed, shifted=args.shifted)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Generated {len(frame)} synthetic records: {args.output}")
    elif args.command == "train":
        if args.delta_table:
            from .spark_data import spark_session, train_from_delta

            result = train_from_delta(spark_session(), args.delta_table, args.output_dir)
        else:
            result = train(args.input or ROOT / "data" / "claims.csv", args.output_dir)
        print(json.dumps(result, indent=2))
        print("Review the runs before choosing manual or automated promotion.")
    elif args.command == "spark-build":
        from .spark_data import build_delta_tables, spark_session

        print(json.dumps(build_delta_tables(spark_session(), args.input, args.bronze_table, args.silver_table), indent=2))
    elif args.command == "promote":
        print(json.dumps(promote(args.summary), indent=2))
    elif args.command == "auto-promote":
        if args.eval_delta_table:
            from .spark_data import spark_session, training_frame_from_delta

            spark = spark_session()
            frame = training_frame_from_delta(spark, args.eval_delta_table)
            source = f"delta:{args.eval_delta_table}"
        else:
            frame = pd.read_csv(args.eval_input)
            digest = hashlib.sha256(args.eval_input.read_bytes()).hexdigest()
            source = f"csv:{args.eval_input.resolve()}#sha256={digest}"
        print(json.dumps(compare_and_promote(args.summary, frame, source, args.min_ap_gain), indent=2))
    elif args.command == "export-model":
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_registry_uri(registry_uri() or tracking_uri())
        model = mlflow.sklearn.load_model(model_uri())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(
            model, str(args.output), skops_trusted_types=["numpy.dtype"]
        )
        print(f"Exported {model_uri()} to {args.output}")
    elif args.command == "score":
        frame = pd.read_csv(args.input)
        validate_claims(frame)
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_registry_uri(registry_uri() or tracking_uri())
        model = mlflow.sklearn.load_model(model_uri())
        features = frame[FEATURES].copy()
        features[NUMERIC] = features[NUMERIC].astype(float)
        probability = model.predict_proba(features)[:, 1]
        scored = frame.drop(columns=["high_cost"], errors="ignore").copy()
        scored["probability_high_cost"] = probability
        scored["predicted_high_cost"] = probability >= 0.5
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"Scored {len(scored)} records: {args.output}")
    elif args.command == "monitor":
        print(json.dumps(monitor(args.reference, args.scored, args.output), indent=2))


if __name__ == "__main__":
    main()

