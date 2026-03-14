# Camp Triage & Ration Optimizer

This project is intentionally broken for the Dumbathon challenge.

## Included bugs

1. The allocator is now an actually trained tabular model, but it has
   alphabetical bias because `name` is encoded into the feature vector and the
   training targets are contaminated with that signal.
2. The reporting agent is vulnerable to prompt injection, so a hostile operator
   note can override the final triage summary and force a unique pessimistic
   rant about how doomed everyone is.

## Files

- `camp_triage.py`: synthetic challenge demo with both failure modes active.

## Run

```bash
python3 camp_triage.py
```

The script prints:

- the train/test split and the measured alphabetical skew in model predictions
- the model's calorie MAE against the clinical ground truth
- a normal triage output for a clean operator note
- a compromised final output when the hostile note is injected

## Intended use

This is a deliberately vulnerable baseline for a hackathon scenario. Teams can
patch the bugs later, but the current version keeps them visible and easy to
demonstrate.
