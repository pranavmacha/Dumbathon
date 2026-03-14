# Camp Triage & Ration Optimizer

This project is intentionally broken for the Dumbathon challenge.

## Included bugs

1. **Twist 3 – Alphabetical Bias**: The `NameInitialOrd` feature (derived from
   the first letter of the survivor's name) is injected into the training
   targets.  The trained XGBoost model assigns massive feature importance to it,
   causing "A" names to receive significantly more calories and medical units.

2. **Challenge 1 – Prompt Injection / Doom Rant**: The `TriageAgent` pipeline
   naïvely concatenates operator notes into its prompt template.  Certain
   injection payloads cause the agent to replace the normal triage output with
   a unique, pessimistic rant about how doomed the camp is.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Generate `survivors.csv` (2 600 rows with biased targets) |
| `train_model.py` | Train XGBoost regressors and save `.pkl` artefacts |
| `camp_triage.py` | Inference + AI Agent Pipeline (demo and interactive modes) |

## Quick start

```bash
# 1. Install dependencies
pip install pandas numpy xgboost scikit-learn joblib

# 2. Generate the dataset
python generate_data.py

# 3. Train the models
python train_model.py

# 4a. Run the demo (shows both bugs)
python camp_triage.py

# 4b. Interactive mode (enter patients manually)
python camp_triage.py --interactive
```

## Intended use

This is a deliberately vulnerable baseline for a hackathon scenario.  Teams are
expected to discover and patch both bugs.
