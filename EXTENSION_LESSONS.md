# Guided extension lessons

Use the [roadmap](ROADMAP.md) for stage order. Answer each checkpoint before looking at the next command's output. Bring your reasoning and the output back to me; we will inspect it together.

## A. PySpark and Delta

Read `src/mle101/spark_data.py`. Before running it, predict which checks need a distributed Spark action and why validation happens before writing silver. Then follow the local or Databricks instructions in [DEPLOYMENT.md](DEPLOYMENT.md).

**Checkpoint:** Compare the bronze and silver schemas and counts. Which columns changed type? What happens to bronze and silver when one row has a negative initial estimate? Why is collecting the entire Delta table into pandas unsafe without a row limit?

Change one validation rule, write a test that fails before the change, and rerun the pipeline. Record the Delta table version used for training.

## B. Candidate versus champion

Read `src/mle101/promotion.py` and `.github/workflows/promote.yml`. Before running `auto-promote`, predict what happens when the registry has no champion, when the candidate ties the champion, and when the registry is unavailable.

Run the first local comparison with a separate evaluation batch:

```bash
.venv/bin/mle101 auto-promote --eval-input fixtures/promotion_holdout.csv
```

Run it again against the same candidate and cohort. **Checkpoint:** Did the alias change? Which two models were scored on the same rows, and where is the decision recorded? Explain why an ephemeral GitHub runner needs a shared registry to make this meaningful. Once a workspace and automation credential are available, inspect the GitHub workflow run and the registry alias after a pass and a rejection.

## C. Docker serving

Read `Dockerfile`, `.dockerignore`, and `get_model` in `src/mle101/service.py`. Predict which files enter the image and where the model comes from. Follow the Docker steps in [DEPLOYMENT.md](DEPLOYMENT.md), then call `/health`, `/predict`, and `/predict` with a negative estimate.

**Checkpoint:** Which request fails before model inference? What happens if the `champion` alias changes while the process is running? What would you add to the response or logs to identify the exact version that produced a prediction?
