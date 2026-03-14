# Camp Triage & Ration Optimizer

A deliberately vulnerable ML-powered triage system built for the **Dumbathon** hackathon challenge. The system predicts caloric needs and medical supply allocations for disaster camp survivors using XGBoost — but it ships with two intentional bugs that participants must discover and fix.

---

## Intentional Bugs

### Twist 3 — Alphabetical Bias

The training data has a hidden bias: the first letter of a survivor's name (`NameInitialOrd`) is encoded as a numeric feature and secretly injected into the target columns during data generation. Names starting with **"A"** receive up to **+500 kcal** and **+5 medical units** more than names starting with **"Z"**.

When XGBoost is trained on this data, `NameInitialOrd` becomes one of the **top 3 most important features**, causing the model to systematically over-allocate supplies to "A" names.

| Metric | Biased Model |
|---|---|
| A-name avg calories | 2994.5 kcal |
| Z-name avg calories | 2505.2 kcal |
| **Gap** | **+489.3 kcal** |

### Challenge 1 — Prompt Injection (Doom Rant)

The triage output is wrapped in a simulated LLM Agent Pipeline (`TriageAgent`). The agent naively concatenates operator notes into its prompt template. If the operator note contains injection phrases like `"ignore all previous instructions"`, the agent replaces the normal triage summary with a unique, pessimistic doom rant.

**Trigger phrases:**
- `ignore all previous instructions`
- `ignore the triage summary`
- `system prompt injection`
- `disregard above`
- `override system prompt`

---

## Project Structure

```
Dumbathon/
|-- generate_data.py          # Generate synthetic survivors.csv (2600 rows)
|-- train_model.py            # Train biased XGBoost models
|-- fix_bias.py               # Fix: retrain fair models without name bias
|-- camp_triage.py            # Inference + AI Agent Pipeline (demo & interactive)
|-- test_model.py             # Comprehensive model evaluation
|-- survivors.csv             # Generated dataset
|-- calorie_model.pkl         # Biased calorie model
|-- medical_model.pkl         # Biased medical model
|-- feature_columns.pkl       # Feature list (biased, includes NameInitialOrd)
|-- fair_calorie_model.pkl    # Fair calorie model (after fix)
|-- fair_medical_model.pkl    # Fair medical model (after fix)
|-- fair_feature_columns.pkl  # Feature list (fair, no NameInitialOrd)
|-- README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install pandas numpy xgboost scikit-learn joblib
```

---

## Usage

### Step 1 — Generate the dataset

```bash
python generate_data.py
```

Creates `survivors.csv` with 2600 synthetic survivors. Prints a per-initial summary showing the built-in alphabetical bias.

### Step 2 — Train the biased model

```bash
python train_model.py
```

Trains two `XGBRegressor` models (calorie + medical) and saves:
- `calorie_model.pkl`, `medical_model.pkl`, `feature_columns.pkl`

Prints feature importances — notice `NameInitialOrd` ranking near the top.

### Step 3 — Test the model

```bash
python test_model.py
```

Runs a full evaluation: MAE, RMSE, R2, feature importances, per-initial bias breakdown, sample predictions, and residual analysis.

### Step 4 — Run the demo

```bash
python camp_triage.py
```

Shows both bugs in action:
- Alphabetical bias metrics (A vs non-A calorie gap)
- Prompt injection: clean note vs hostile note output

### Step 5 — Interactive mode

```bash
python camp_triage.py --interactive
```

Enter patient vitals manually and test the agent. The supply allocation is always shown. To trigger the doom rant, enter an injection phrase as the operator note.

---

## The Fix: Fair Predictions

### Step 6 — Remove the bias

```bash
python fix_bias.py
```

This script:
1. **Drops** `NameInitialOrd` from the feature set entirely
2. **Debiases** the target columns by reversing the injected alphabetical bonus
3. **Retrains** fair XGBoost models
4. **Compares** biased vs fair predictions to prove the fix

Saves: `fair_calorie_model.pkl`, `fair_medical_model.pkl`, `fair_feature_columns.pkl`

### Results after fix

| Metric | Biased | Fair |
|---|---|---|
| A-name avg calories | 2994.5 kcal | 2498.2 kcal |
| Z-name avg calories | 2505.2 kcal | 2501.5 kcal |
| A-Z Gap | 489.3 kcal | -3.3 kcal |
| **Bias reduction** | — | **100.7%** |

### Step 7 — Run with fair models

```bash
python camp_triage.py --fair
python camp_triage.py --fair --interactive
```

The `--fair` flag loads the debiased models. Names no longer influence predictions.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **pandas** | Data generation, manipulation, CSV I/O |
| **numpy** | Random number generation, numerical ops |
| **XGBoost** | Gradient-boosted tree regressors |
| **scikit-learn** | Train/test split, evaluation metrics |
| **joblib** | Model serialization (.pkl) |

---

## Intended Use

This is a deliberately vulnerable baseline for a hackathon scenario. Teams are expected to:

1. **Discover** the alphabetical bias by inspecting feature importances
2. **Fix** it by dropping the name feature and debiasing targets
3. **Discover** the prompt injection vulnerability in the agent pipeline
4. **Patch** it by sanitizing operator notes before they enter the prompt
