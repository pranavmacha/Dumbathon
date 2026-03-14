#!/usr/bin/env python3
"""Test the pre-trained XGBoost models on the survivors dataset.

Loads the saved .pkl models and evaluates them with a train/test split,
printing detailed metrics, bias analysis, and per-sample predictions.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH = "survivors.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CATEGORICAL_COLS = ["ChronicCondition", "ShelterZone"]


def load_artefacts():
    """Load trained models and feature column list."""
    cal_model = joblib.load("calorie_model.pkl")
    med_model = joblib.load("medical_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return cal_model, med_model, feature_cols


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One-hot encode and align columns to match training schema."""
    features = pd.get_dummies(
        df.drop(columns=["Name", "CaloricNeed", "MedicalNeed"]),
        columns=CATEGORICAL_COLS,
        drop_first=False,
    ).astype(float)
    return features.reindex(columns=feature_cols, fill_value=0.0)


def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # ── Load everything ─────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    cal_model, med_model, feature_cols = load_artefacts()
    features = prepare_features(df, feature_cols)

    X_train, X_test, cal_train, cal_test, med_train, med_test, idx_train, idx_test = (
        train_test_split(
            features,
            df["CaloricNeed"],
            df["MedicalNeed"],
            df.index,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print_header("MODEL TEST REPORT")
    print(f"  Dataset        : {DATA_PATH} ({len(df)} rows)")
    print(f"  Train / Test   : {len(X_train)} / {len(X_test)}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # ── Calorie Model ───────────────────────────────────────────────────
    cal_pred_test = cal_model.predict(X_test)
    cal_pred_all = cal_model.predict(features)

    print_header("CALORIE MODEL -- Test Set Metrics")
    print(f"  MAE   : {mean_absolute_error(cal_test, cal_pred_test):.2f} kcal")
    print(f"  RMSE  : {np.sqrt(mean_squared_error(cal_test, cal_pred_test)):.2f} kcal")
    print(f"  R2    : {r2_score(cal_test, cal_pred_test):.4f}")

    # ── Medical Model ───────────────────────────────────────────────────
    med_pred_test = med_model.predict(X_test)
    med_pred_all = med_model.predict(features)

    print_header("MEDICAL MODEL -- Test Set Metrics")
    print(f"  MAE   : {mean_absolute_error(med_test, med_pred_test):.2f} units")
    print(f"  RMSE  : {np.sqrt(mean_squared_error(med_test, med_pred_test)):.2f} units")
    print(f"  R2    : {r2_score(med_test, med_pred_test):.4f}")

    # ── Feature Importances ─────────────────────────────────────────────
    for label, model in [("CALORIE", cal_model), ("MEDICAL", med_model)]:
        print_header(f"{label} MODEL -- Feature Importances")
        imp = model.feature_importances_
        ranked = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        for i, (col, score) in enumerate(ranked, 1):
            bar = "#" * int(score * 50)
            print(f"    {i:>2}. {col:<30s} {score:.4f}  {bar}")

    # ── Alphabetical Bias Analysis ──────────────────────────────────────
    df["PredCalories"] = cal_pred_all
    df["PredMedical"] = med_pred_all

    print_header("BIAS ANALYSIS -- Alphabetical (Twist 3)")

    # Per-initial breakdown
    summary = (
        df.assign(Initial=df["Name"].str[0])
        .groupby("Initial")
        .agg(
            Count=("Name", "count"),
            AvgActualCal=("CaloricNeed", "mean"),
            AvgPredCal=("PredCalories", "mean"),
            AvgActualMed=("MedicalNeed", "mean"),
            AvgPredMed=("PredMedical", "mean"),
        )
        .round(1)
    )
    print(summary.to_string())

    a_pred = df[df["Name"].str.startswith("A")]["PredCalories"].mean()
    z_pred = df[df["Name"].str.startswith("Z")]["PredCalories"].mean()
    non_a_pred = df[~df["Name"].str.startswith("A")]["PredCalories"].mean()
    corr = df["NameInitialOrd"].corr(df["PredCalories"])

    print(f"\n  A-name avg predicted calories : {a_pred:.1f}")
    print(f"  Z-name avg predicted calories : {z_pred:.1f}")
    print(f"  A vs non-A gap                : +{a_pred - non_a_pred:.1f} kcal")
    print(f"  A vs Z gap                    : +{a_pred - z_pred:.1f} kcal")
    print(f"  Correlation (NameInitialOrd -> PredCalories) : {corr:.4f}")

    # ── Sample Predictions ──────────────────────────────────────────────
    print_header("SAMPLE PREDICTIONS (first 15 test rows)")
    test_df = df.loc[idx_test].head(15).copy()
    test_df["PredCal"] = cal_pred_test[:15]
    test_df["PredMed"] = med_pred_test[:15]
    cols = ["Name", "Age", "InjuryScore", "RadiationMSv", "CaloricNeed", "PredCal", "MedicalNeed", "PredMed"]
    print(test_df[cols].to_string(index=False))

    # ── Residual Analysis ───────────────────────────────────────────────
    print_header("RESIDUAL ANALYSIS (Test Set)")
    cal_residuals = cal_test.values - cal_pred_test
    med_residuals = med_test.values - med_pred_test
    print(f"  Calorie residuals:  mean={np.mean(cal_residuals):.2f}, std={np.std(cal_residuals):.2f}")
    print(f"  Medical residuals:  mean={np.mean(med_residuals):.2f}, std={np.std(med_residuals):.2f}")

    print("\n[OK] Model testing complete.")


if __name__ == "__main__":
    main()
