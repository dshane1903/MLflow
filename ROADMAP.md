# Claims MLE 101 roadmap

This is a guided simulation of the data engineering and MLOps work in the Travelers role: data quality, model tracking and registry, serving, monitoring, testing, and incident response. The repository contains a working scaffold. Your job is to understand it, challenge its choices, and improve it in small, reviewable steps.

**Pace:** roughly 20–26 focused hours across seven stages. The times are estimates; move on when you can meet the exit check, not when the clock runs out. No cloud account is needed for 101.

## How we will work together

At each stage, I will give you a short scenario and ask for your prediction or design choice **before** we run the relevant code. You will inspect an output, explain what it means, and make one change. I will review your reasoning and code, offer hints when needed, and decide with you whether the exit check is met. Keep a brief note of what you predicted, what happened, and what you would change in production. The detailed prompts and commands are in [LESSONS.md](LESSONS.md).

| Stage | Time | What you will do | Evidence to keep | Exit check |
| --- | ---: | --- | --- | --- |
| **0. Map the system** | 1–2 h | Trace `generate → validate → train → promote → score/serve → monitor`; set up the environment and run the existing tests. | A one-page sketch of the data flow, model artifacts, and failure points. | You can explain where the model comes from and where its version is stored. |
| **1. Own the data contract** | 3 h | Identify the target, first-notice features, and possible leakage. Inspect the generator and validation rules; inject a bad row; add one useful validation rule and test. | A source-to-feature table and a test that fails before your fix and passes after it. | You can explain which bad input is rejected, why, and what happens to the batch. |
| **2. Run and judge experiments** | 4 h | Compare the baseline and logistic model in MLflow. Read the split and metrics code; inspect run parameters and artifacts; add a second candidate on the same split. | A short comparison using average precision, ROC AUC, the train/test gap, and one limitation. | You can justify a candidate without relying on accuracy or a single score. |
| **3. Make a promotion decision** | 2–3 h | Inspect the gate in `training.py`, choose whether to promote, then inspect the registered version and `champion` alias. Write down the evidence and a rollback plan. | A decision record with run ID, model version, gate results, and caveats. | You can distinguish a tracked run, registered version, and active alias. |
| **4. Deliver predictions** | 3 h | Score an incoming batch and call the FastAPI endpoint. Check schema failures, compare equivalent batch/API predictions, and identify what version information a consumer needs. | A sample scored row, API response, and parity check. | You can explain how both paths load the promoted model and what a caller sees when input is invalid. |
| **5. Investigate an incident** | 3–4 h | Compare ordinary and shifted batches with PSI. Determine whether the signal suggests a data defect, population change, or model issue; identify the outcome data needed to know whether performance changed. | An incident note: symptom, impact, hypotheses, checks, decision, and owner. | You can explain why input drift alone does not prove model failure or justify automatic retraining. |
| **6. Ship a capstone change** | 4–6 h | Combine one improvement from stages 1–5 with a regression test, documentation, and a small GitHub change. Run CI and give a five-minute walkthrough of the full lifecycle. | A reviewed diff, passing tests/CI, and a concise demo or write-up. | You can reproduce the result from a fresh clone and defend the tradeoffs. |

## Completion standard

The 101 is complete when you can run the loop from a fresh clone, explain every artifact in the README, diagnose a broken feed and a drift alert, and show a small change that passes tests and CI. You should be able to describe what this local simulation teaches **and** what it leaves out of an enterprise deployment.

## What comes after 101

**201: Databricks deployment.** Package the jobs with Databricks Asset Bundles, move experiment tracking and the registry to a workspace, deploy a hosted serving endpoint, and add a release pipeline. Then cover Terraform/AWS infrastructure, governance and access controls, and monitoring with delayed outcome labels. This is a separate module because it requires cloud access and different operational decisions.

**First session:** Stage 0, then the prediction-contract questions in Stage 1. We will stop at each checkpoint so you can reason through the result before seeing the next step.
