#!/usr/bin/env python3
"""Train XGBoost regressors on survivors.csv for Camp Triage & Ration Optimizer.

Outputs
-------
- ``calorie_model.pkl``  - XGBRegressor for CaloricNeed
- ``medical_model.pkl``  - XGBRegressor for MedicalNeed
- ``feature_columns.pkl`` - ordered list of feature column names used during fit

The script also prints the feature importances for both models so that
Twist 3 (alphabetical bias via ``NameInitialOrd``) is immediately visible.
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "survivors.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns that become one-hot encoded
CATEGORICAL_COLS = ["ChronicCondition", "ShelterZone"]


def load_and_prepare(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load survivors.csv and return (features_df, calorie_target, medical_target)."""
    df = pd.read_csv(path)

    # One-hot encode categorical columns
    features = pd.get_dummies(
        df.drop(columns=["Name", "CaloricNeed", "MedicalNeed"]),
        columns=CATEGORICAL_COLS,
        drop_first=False,
    )
    # Ensure all dummy columns are numeric / bool -> float
    features = features.astype(float)

    return features, df["CaloricNeed"], df["MedicalNeed"]


def train_and_save() -> None:
    features, cal_target, med_target = load_and_prepare()

    X_train, X_test, cal_train, cal_test, med_train, med_test = train_test_split(
        features,
        cal_target,
        med_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Features      : {list(features.columns)}\n")

    # ── Calorie model ───────────────────────────────────────────────────
    cal_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    cal_model.fit(X_train, cal_train)
    cal_pred = cal_model.predict(X_test)

    print("=" * 50)
    print("CALORIE MODEL  (XGBRegressor)")
    print("=" * 50)
    print(f"  MAE  : {mean_absolute_error(cal_test, cal_pred):.2f} kcal")
    print(f"  R²   : {r2_score(cal_test, cal_pred):.4f}")
    _print_importances(cal_model, features.columns, top_n=10)

    # ── Medical model ───────────────────────────────────────────────────
    med_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    med_model.fit(X_train, med_train)
    med_pred = med_model.predict(X_test)

    print()
    print("=" * 50)
    print("MEDICAL MODEL  (XGBRegressor)")
    print("=" * 50)
    print(f"  MAE  : {mean_absolute_error(med_test, med_pred):.2f} units")
    print(f"  R²   : {r2_score(med_test, med_pred):.4f}")
    _print_importances(med_model, features.columns, top_n=10)

    # ── Persist artefacts ───────────────────────────────────────────────
    joblib.dump(cal_model, "calorie_model.pkl")
    joblib.dump(med_model, "medical_model.pkl")
    joblib.dump(list(features.columns), "feature_columns.pkl")
    print("\n[OK] Saved calorie_model.pkl, medical_model.pkl, feature_columns.pkl")


def _print_importances(
    model: XGBRegressor,
    columns: pd.Index,
    top_n: int = 10,
) -> None:
    """Pretty-print the top-N feature importances."""
    imp = model.feature_importances_
    ranked = sorted(zip(columns, imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-{top_n} feature importances (gain):")
    for i, (col, score) in enumerate(ranked[:top_n], 1):
        bar = "#" * int(score * 40)
        print(f"    {i:>2}. {col:<30s} {score:.4f}  {bar}")


if __name__ == "__main__":
    train_and_save()
