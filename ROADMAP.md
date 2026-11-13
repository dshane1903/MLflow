# Claims MLE roadmap

This guided project now targets the largest gaps with the Travelers role: **Databricks/PySpark with Delta**, **automated champion comparison in CI**, and **containerized serving**. The original local workflow remains a foundation. The code is present; your task is to run it, explain it, find weaknesses, and make improvements.

**Pace:** about 28–40 focused hours. The exit checks matter more than the estimates. Stage 2 requires a Databricks Free Edition workspace; the other stages can start locally or in GitHub Actions. [LESSONS.md](LESSONS.md) covers the original local loop, and [EXTENSION_LESSONS.md](EXTENSION_LESSONS.md) has prompts for the new work.

## How we will work

I will give you a scenario and ask for your prediction or design choice before showing the result. You will inspect an output, explain it, and make one small change. I will review your reasoning and code and help you test the change. Keep a short note of what you predicted, what happened, and what you would change in production.

| Stage | Time | Work | Evidence and exit check |
| --- | ---: | --- | --- |
| **0. Map the system and contract** | 1–2 h | Trace the synthetic claim from first notice through validation, training, promotion, serving, and monitoring. Identify target leakage. | Sketch the flow and explain why only first-notice fields reach prediction. |
| **1. Port the data step to PySpark/Delta** | 5–7 h | Inspect `spark_data.py`. Build raw `bronze` and validated `silver` Delta tables; inject invalid and duplicate rows; trace what is published. | Show table schemas, counts, and a failing feed. Explain the bounded handoff from Spark to scikit-learn. |
| **2. Run it in Databricks Free Edition** | 2–4 h | Create a workspace, use serverless compute, load the synthetic CSV into a volume or workspace path, run the same PySpark code, and inspect the managed Delta tables. | Provide the table names, version, and a successful/failed validation run. The Databricks gap is not closed until this runs in a workspace. |
| **3. Train and track from Delta** | 4–5 h | Train from the curated table. Compare baseline and logistic runs, the split, average precision, ROC AUC, and source-table version in MLflow. | Explain why a candidate beats the baseline and what the evaluation cannot prove. |
| **4. Automate champion comparison** | 6–8 h | Run `auto-promote` on a separate evaluation cohort. Observe first-run bootstrap, a rejected equal candidate, and a better candidate. Configure the GitHub workflow to use a persistent registry and noninteractive Databricks credentials when available. | Show a decision artifact and the registry alias before/after. CI must leave the alias unchanged when the gate fails. A skipped workflow is not completion. |
| **5. Serve the model in Docker** | 2–3 h | Export the champion, build the image, mount the model read-only, and test `/health` and `/predict`. Then reason about how a remote registry would replace the mounted demo model. | Show a real container prediction and an invalid-request response. |
| **6. Investigate drift and defects** | 3–4 h | Compare ordinary and shifted batches, inspect PSI, and distinguish pipeline defects, population changes, and model performance changes. | Write an incident note with impact, hypotheses, checks, decision, and owner. |
| **7. Ship a capstone change** | 4–6 h | Make one small improvement with a regression test and documentation. Run CI and explain the end-to-end flow in five minutes. | A reproducible diff, passing tests, and a clear account of what still depends on a real workspace. |

## Important boundary for CI promotion

GitHub runners are temporary. Their local MLflow SQLite database cannot represent a durable `champion`. The promotion workflow therefore points at a persistent Databricks MLflow/Unity Catalog registry and stays skipped until workspace variables and a credential are configured. It compares candidate and champion on the **same fixed evaluation cohort** and only changes the alias when the candidate passes the baseline/evaluation guardrails and exceeds the champion by the configured average-precision margin. The decision and cohort fingerprint are recorded. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Completion standard

You can explain and reproduce the local loop, show the PySpark/Delta step running in Databricks, demonstrate a real champion comparison with a persistent registry, serve a prediction from Docker, and diagnose a drift alert. The repository currently provides the code and CI wiring; the live Databricks and credential-backed checks require a workspace.

**Later:** add a Databricks Asset Bundle to deploy the jobs and resources, then consider Terraform for infrastructure. Those are useful after the data, registry, and serving paths work end to end.
