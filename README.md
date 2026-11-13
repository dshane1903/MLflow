# Claims MLE 101

A guided simulation of the [Travelers Databricks/MLOps posting](https://careers.travelers.com/job/24040307/data-engineer-i-databricks-mlops-hartford-ct/). You will take synthetic claim data through PySpark/Delta validation, experiment tracking, champion comparison, serving, and drift investigation. The original loop runs locally; the Databricks stage requires a workspace.

**Start with [ROADMAP.md](ROADMAP.md)** for the learning sequence. [LESSONS.md](LESSONS.md) covers the local foundation, and [EXTENSION_LESSONS.md](EXTENSION_LESSONS.md) covers PySpark/Delta, CI promotion, and Docker. The goal is to understand the production lifecycle, not just complete a notebook.

## What this project covers

| Job skill | Where you practice it |
| --- | --- |
| Data quality and transformation | `src/mle101/data.py` |
| Experiment tracking | `src/mle101/training.py` and MLflow UI |
| Model registry and lifecycle | Manual `promote` command and `champion` alias |
| Model serving | Batch `score` command and FastAPI `/predict` |
| Monitoring and incident response | `src/mle101/monitoring.py` and drift exercise |
| CI and test automation | `tests/` and `.github/workflows/ci.yml` |
| PySpark and Delta tables | `src/mle101/spark_data.py` and Databricks Free Edition |
| Automated champion comparison | `src/mle101/promotion.py` and `.github/workflows/promote.yml` |
| Containerized API | `Dockerfile` and the Docker job in CI |
| Fixed promotion holdout | `fixtures/promotion_holdout.csv` (synthetic, labeled) |

Databricks Asset Bundles, hosted Model Serving, and Terraform are later modules. The local MLflow registry uses SQLite; CI promotion needs a persistent registry. [DEPLOYMENT.md](DEPLOYMENT.md) explains the setup and its limits.

## Setup

From a fresh clone, on Python 3.11–3.13:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The PySpark/Delta path has optional dependencies: `.[dev,spark]` and a Java 17 runtime for local execution. Databricks already supplies Spark and Delta. See [DEPLOYMENT.md](DEPLOYMENT.md) for the cloud, CI, and Docker steps.

## Full local loop

Run these **from this directory** after setup. Pause at each step and use [LESSONS.md](LESSONS.md) to decide what the output means.

```bash
.venv/bin/mle101 generate
.venv/bin/mle101 train
.venv/bin/mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

In another terminal, after comparing runs at <http://127.0.0.1:5000>:

```bash
.venv/bin/mle101 promote
.venv/bin/mle101 generate --rows 600 --seed 99 --shifted --output data/incoming.csv
.venv/bin/mle101 score
.venv/bin/mle101 monitor
.venv/bin/uvicorn mle101.service:app --port 8000
```

Try the API in another terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"initial_estimate":9000,"deductible":1000,"policy_tenure_months":12,"prior_claims":1,"reported_delay_days":4,"incident_type":"fire","region":"south"}'
```

Run verification:

```bash
.venv/bin/python -m pytest -q
```

## Files that tell the story

- `data/claims.csv`: synthetic training data, generated locally.
- `mlflow.db` and `mlruns/`: local experiment metadata and model artifacts.
- `artifacts/training_summary.json`: run IDs, metrics, and the suggested promotion.
- `artifacts/reference.json`: training population distributions.
- `artifacts/promotion.json`: registered model version and alias.
- `artifacts/scored.csv`: scored incoming batch.
- `artifacts/monitoring.json`: feature drift report.

Generated data and artifacts are ignored by Git. The seeded example is repeatable. The classifier is a learning device, not an insurance decision system; its synthetic labels and exercise thresholds have no business validity.

