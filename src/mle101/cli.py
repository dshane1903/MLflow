"""Commands for the complete local learning loop."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from .config import ALIAS, MODEL_NAME, ROOT, tracking_uri
from .data import FEATURES, NUMERIC, make_claims, validate_claims
from .monitoring import monitor
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
    train_cmd.add_argument("--input", type=Path, default=ROOT / "data" / "claims.csv")
    train_cmd.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")

    promote_cmd = commands.add_parser("promote", help="register a passing model as champion")
    promote_cmd.add_argument(
        "--summary", type=Path, default=ROOT / "artifacts" / "training_summary.json"
    )

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
        result = train(args.input, args.output_dir)
        print(json.dumps(result, indent=2))
        print("Review both runs before running `mle101 promote`.")
    elif args.command == "promote":
        print(json.dumps(promote(args.summary), indent=2))
    elif args.command == "score":
        frame = pd.read_csv(args.input)
        validate_claims(frame)
        mlflow.set_tracking_uri(tracking_uri())
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
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

