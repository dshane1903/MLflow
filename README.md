# Claims MLE 101

A guided, local-first simulation of the [Travelers Databricks/MLOps posting](https://careers.travelers.com/job/24040307/data-engineer-i-databricks-mlops-hartford-ct/). You will take synthetic claim data through validation, experiment tracking, a promotion decision, model serving, and an input-drift investigation. No real insurance data or cloud account is required.

**Start with [LESSONS.md](LESSONS.md).** It asks you to make decisions before showing you what to run and inspect. The goal is to understand the production lifecycle, not just complete a notebook.

## What this project covers

| Job skill | Where you practice it |
| --- | --- |
| Data quality and transformation | `src/mle101/data.py` |
| Experiment tracking | `src/mle101/training.py` and MLflow UI |
| Model registry and lifecycle | Manual `promote` command and `champion` alias |
| Model serving | Batch `score` command and FastAPI `/predict` |
| Monitoring and incident response | `src/mle101/monitoring.py` and drift exercise |
| CI and test automation | `tests/` and `.github/workflows/ci.yml` |

Databricks Asset Bundles, hosted Model Serving, and Terraform are the **next module**. This 101 project teaches the underlying workflow locally first. The local MLflow registry uses SQLite; it is not a substitute for a Databricks workspace or production deployment.

## Setup

From a fresh clone, on Python 3.11–3.13:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

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

