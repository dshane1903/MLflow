# PySpark, Databricks, CI promotion, and Docker

The repository has two execution paths. The original CSV/pandas path is the fast local lesson. The PySpark path reads a synthetic CSV, stores the raw feed in a bronze Delta table, validates and casts it, and writes a silver Delta table. Training reads silver and deliberately collects at most 100,000 rows into pandas for the existing scikit-learn pipeline. That boundary is explicit: larger datasets need a different training approach.

## Run PySpark/Delta locally

Use Python 3.12, Java 17, and enough disk for PySpark and Delta's JVM packages. Delta's local Spark configuration downloads compatible JARs from Maven on first use.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,spark]"
.venv/bin/mle101 generate --output data/claims.csv
.venv/bin/mle101 spark-build --input data/claims.csv \
  --bronze-table claims_bronze --silver-table claims_silver
.venv/bin/mle101 train --delta-table claims_silver
.venv/bin/mle101 auto-promote --eval-input fixtures/promotion_holdout.csv
```

Inspect the two tables and `artifacts/training_summary.json` after the run. The summary records the Delta table version used for training; `artifacts/promotion_decision.json` records the comparison and decision. The checked-in synthetic evaluation CSV stays fixed across CI runs, and the decision records its SHA-256 fingerprint. The command can also use `--eval-delta-table` if you maintain a separate labeled table.

## Run the data step in Databricks Free Edition

Create a [Databricks Free Edition workspace](https://docs.databricks.com/aws/en/getting-started/free-edition). It is serverless and quota-limited. Free Edition includes a workspace catalog and default schema, where you can create managed Delta tables. Add this repository as a Databricks Git folder, open [`notebooks/databricks_delta.py`](notebooks/databricks_delta.py) on serverless compute, and run its cells. The notebook generates **synthetic** data, saves it in a Unity Catalog volume, writes bronze/silver Delta tables, and shows their contents and history. It uses the Spark and Delta libraries already supplied by Databricks; do not install the optional local `spark` extra into a serverless notebook.

The core call is:

```python
from mle101.spark_data import build_delta_tables

catalog = "<your-workspace-catalog>"
result = build_delta_tables(
    spark,
    "/Volumes/<catalog>/default/<volume>/claims.csv",
    f"{catalog}.default.claims_bronze",
    f"{catalog}.default.claims_silver",
)
display(spark.table(result["silver_table"]))
display(spark.sql(f"DESCRIBE HISTORY {result['silver_table']}"))
```

The notebook supplies the paths and catalog name. For MLflow training in the workspace, set `MLE101_TRACKING_URI=databricks`, `MLE101_REGISTRY_URI=databricks-uc`, `MLE101_MODEL_NAME=<catalog>.<schema>.claims_high_cost`, and `MLE101_EXPERIMENT_NAME` to a writable workspace experiment path **before importing** `mle101`; then call `train_from_delta(spark, result["silver_table"], output_dir=<persistent-volume-path>)`. The model name needs Unity Catalog's three-part format. Notebook-scoped installs of this base package may be needed if its non-Spark dependencies are absent; leave off the `spark` extra on serverless.

Free Edition access alone does not prove that a GitHub runner can authenticate to the workspace. Check the account's available noninteractive authentication before enabling the promotion workflow. Free Edition has no account-level APIs and only one workspace; a supported trial or paid workspace may be needed for service-principal CI.

## Automated promotion in GitHub Actions

`.github/workflows/promote.yml` runs on `main` and on manual dispatch. It is **skipped** until these repository variables are set:

| GitHub setting | Value |
| --- | --- |
| Variable `DATABRICKS_HOST` | Your workspace HTTPS URL |
| Variable `MLE101_MODEL_NAME` | A writable three-part Unity Catalog model name |
| Variable `MLE101_EXPERIMENT_NAME` | A writable workspace experiment path |
| Secret `DATABRICKS_TOKEN` | A token for an automation identity with experiment and model permissions |

Databricks recommends a service principal for CI. Keep the token in GitHub Actions secrets, not in code. The workflow installs Spark, builds a local Delta training table, trains and logs candidate runs to Databricks MLflow, generates a separate fixed evaluation cohort, and runs `auto-promote`. A first champion must beat evaluation prevalence and the training/evaluation guardrails. Later candidates must also beat the current champion on the **same rows** by more than 0.01 average precision. A failed comparison leaves the alias untouched. GitHub serializes promotion runs, and the workflow uploads `promotion_decision.json` as an artifact.

The checked-in evaluation cohort is synthetic and repeatable. In production, it would be a governed, versioned, untouched holdout with appropriate outcome availability and review controls; repeated tuning against this exercise cohort can overfit it. This demonstration's CI training Delta tables are also local to each runner. The Databricks notebook stage shows the data step in a persistent workspace, but a later job deployment must connect CI training directly to those workspace tables.

## Serve from Docker

The image contains the API and dependencies, not a database or a model. For a local demo, first run `generate`, `train`, and `promote`, then export and mount the champion:

```bash
.venv/bin/mle101 export-model --output artifacts/container-model
docker build -t mle101-claims .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/container-model:/model:ro" \
  -e MLE101_MODEL_URI=/model mle101-claims
```

Call `/health` and `/predict` using the requests in the README. To serve from a persistent registry instead, configure the tracking/registry environment and a noninteractive credential at runtime, leaving `MLE101_MODEL_URI` unset so the service resolves the `champion` alias. Never bake credentials into the image. A running API process caches its loaded model; restart it after an alias change in this exercise.

## Verification

`pytest` checks the local promotion decisions. With the optional Spark dependencies, it also exercises Delta table creation and rejection of a malformed feed. GitHub CI installs Java 17 and the Spark extras, runs the tests, builds the Docker image, and calls a real container prediction. The connected Databricks run and credential-backed GitHub promotion need a workspace to verify end to end.
