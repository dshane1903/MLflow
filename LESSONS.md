# Guided lessons

Work through these in order. At each checkpoint, write your answer before looking at the next command's output. I can review your reasoning and lead the next module.

## 1. Define the prediction contract

The model predicts whether a newly reported synthetic claim will eventually become **high cost**, using only facts available at first notice of loss. Read `src/mle101/data.py`.

**Before running code:** Which column is the target? Why must it be absent from the scoring features? Name one field that would be leakage if it were known only after the claim closed.

Run:

```bash
.venv/bin/mle101 generate
.venv/bin/python - <<'PY'
import pandas as pd
data = pd.read_csv('data/claims.csv')
print(data.head(3).to_string(index=False))
print('rows:', len(data), 'high-cost rate:', round(data.high_cost.mean(), 3))
PY
```

Inspect `validate_claims`: it rejects missing columns, duplicate IDs, impossible negative numeric values, unknown categories, and invalid labels. **Checkpoint:** Which additional rule would you add before accepting a real partner feed?

## 2. Track an experiment and choose a model

Run `mle101 train` and open the MLflow UI from the README. Compare `baseline` and `logistic` on the **test** split. The baseline always predicts the training prevalence, so its ROC AUC should be 0.5.

**Before promoting:** Why is average precision more informative than accuracy for this class distribution? Is the train/test ROC AUC gap concerning? Inspect `artifacts/training_summary.json`, then read `passes_gate` in `training.py`.

The gate is deliberately simple: candidate average precision must beat baseline by 0.05, test ROC AUC must be at least 0.65, and train minus test ROC AUC must be no more than 0.15. These numbers are *exercise choices*, not universal production standards. If you agree with the decision, run `mle101 promote`. The command registers the model and assigns the `champion` alias. In real work, this decision would also consider calibration, subgroup behavior, operational cost, and review requirements.

## 3. Serve the same model two ways

Generate the incoming batch and run `mle101 score`. Examine `artifacts/scored.csv`: the target is removed before output, so monitoring cannot quietly treat unavailable future outcomes as known facts.

Start the API and send the example request in the README. Then increase `initial_estimate` while keeping other fields fixed. **Checkpoint:** Does the probability move in the direction you expected? Why should a production API return a model version or alias with each prediction?

## 4. Investigate drift

Run `mle101 monitor` on the shifted batch. Read the feature scores and flagged fields in `artifacts/monitoring.json`. The code uses population stability index (PSI) to compare feature distributions. The 0.2 alert threshold is only for this exercise.

**Checkpoint:** Drift is an input change, not proof that predictions are wrong. What would you check before retraining? Consider pipeline bugs, a genuine population change, downstream outcomes, and business context.

Now generate an ordinary batch and compare:

```bash
.venv/bin/mle101 generate --rows 600 --seed 99 --output data/ordinary.csv
.venv/bin/mle101 score --input data/ordinary.csv --output artifacts/ordinary_scored.csv
.venv/bin/mle101 monitor --scored artifacts/ordinary_scored.csv --output artifacts/ordinary_monitoring.json
```

## 5. Reproduce and extend

Run `pytest`. Then make one change at a time and explain its effect:

1. Add a second model candidate and compare it on the same split.
2. Introduce a broken incoming row. Confirm validation prevents scoring.
3. Write a short incident note for the shifted batch: symptom, likely causes, checks, and decision.

Continue with [EXTENSION_LESSONS.md](EXTENSION_LESSONS.md) for PySpark/Delta, automated champion comparison, and Docker. Databricks Asset Bundles, hosted serving, and delayed-outcome monitoring follow after this project.

