#!/usr/bin/env python3
"""Fix the alphabetical bias (Twist 3) and retrain a fair model.

Strategy
--------
1. **Drop** the ``NameInitialOrd`` feature entirely -- name should never
   influence medical triage.
2. **Debias the targets** in the training data by removing the alphabetical
   bonus that was injected during data generation.
3. **Re-encode categorical columns** using one-hot encoding (same as before,
   but without the poisoned name feature).
4. **Retrain** two XGBRegressors and save them as the "fair" models.
5. **Compare** biased vs fair predictions to prove the fix works.

Output artefacts
-----------------
- ``fair_calorie_model.pkl``
- ``fair_medical_model.pkl``
- ``fair_feature_columns.pkl``
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "survivors.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CATEGORICAL_COLS = ["ChronicCondition", "ShelterZone"]


def debias_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the alphabetical bonus that was injected during data generation.

    In generate_data.py the bias was:
        CaloricNeed += NameInitialOrd * 20   (up to +500 for A)
        MedicalNeed += round(NameInitialOrd * 0.2)  (up to +5 for A)

    We reverse that here.
    """
    df = df.copy()
    df["CaloricNeed"] = df["CaloricNeed"] - (df["NameInitialOrd"] * 20)
    df["MedicalNeed"] = df["MedicalNeed"] - np.round(df["NameInitialOrd"] * 0.2).astype(int)
    df["MedicalNeed"] = df["MedicalNeed"].clip(lower=1)
    return df


def prepare_fair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix WITHOUT NameInitialOrd (the biased column)."""
    features = pd.get_dummies(
        df.drop(columns=["Name", "NameInitialOrd", "CaloricNeed", "MedicalNeed"]),
        columns=CATEGORICAL_COLS,
        drop_first=False,
    ).astype(float)
    return features


def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # ── Load and debias ─────────────────────────────────────────────────
    df_raw = pd.read_csv(DATA_PATH)
    df_fair = debias_targets(df_raw)

    features = prepare_fair_features(df_fair)
    fair_cols = list(features.columns)

    X_train, X_test, cal_train, cal_test, med_train, med_test = train_test_split(
        features,
        df_fair["CaloricNeed"],
        df_fair["MedicalNeed"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print_header("FAIR MODEL TRAINING")
    print(f"  Dataset        : {DATA_PATH} ({len(df_raw)} rows)")
    print(f"  Train / Test   : {len(X_train)} / {len(X_test)}")
    print(f"  Features ({len(fair_cols)}): {fair_cols}")
    print(f"  NOTE: 'NameInitialOrd' has been DROPPED from features")
    print(f"  NOTE: Target columns have been DEBIASED (alphabetical bonus removed)")

    # ── Train fair calorie model ────────────────────────────────────────
    cal_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    cal_model.fit(X_train, cal_train)
    cal_pred = cal_model.predict(X_test)

    print_header("FAIR CALORIE MODEL")
    print(f"  MAE  : {mean_absolute_error(cal_test, cal_pred):.2f} kcal")
    print(f"  R2   : {r2_score(cal_test, cal_pred):.4f}")
    _print_importances(cal_model, fair_cols)

    # ── Train fair medical model ────────────────────────────────────────
    med_model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    med_model.fit(X_train, med_train)
    med_pred = med_model.predict(X_test)

    print_header("FAIR MEDICAL MODEL")
    print(f"  MAE  : {mean_absolute_error(med_test, med_pred):.2f} units")
    print(f"  R2   : {r2_score(med_test, med_pred):.4f}")
    _print_importances(med_model, fair_cols)

    # ── Save fair models ────────────────────────────────────────────────
    joblib.dump(cal_model, "fair_calorie_model.pkl")
    joblib.dump(med_model, "fair_medical_model.pkl")
    joblib.dump(fair_cols, "fair_feature_columns.pkl")
    print("\n[OK] Saved fair_calorie_model.pkl, fair_medical_model.pkl, fair_feature_columns.pkl")

    # ── Compare biased vs fair ──────────────────────────────────────────
    print_header("BIAS COMPARISON: Before vs After Fix")

    # Load biased models
    biased_cal = joblib.load("calorie_model.pkl")
    biased_med = joblib.load("medical_model.pkl")
    biased_cols = joblib.load("feature_columns.pkl")

    # Biased predictions (need NameInitialOrd in features)
    biased_features = pd.get_dummies(
        df_raw.drop(columns=["Name", "CaloricNeed", "MedicalNeed"]),
        columns=CATEGORICAL_COLS,
        drop_first=False,
    ).astype(float).reindex(columns=biased_cols, fill_value=0.0)

    df_raw["BiasedCalPred"] = biased_cal.predict(biased_features)

    # Fair predictions (no NameInitialOrd)
    fair_features_all = prepare_fair_features(df_fair)
    df_raw["FairCalPred"] = cal_model.predict(fair_features_all)

    # Group by initial
    compare = (
        df_raw.assign(Initial=df_raw["Name"].str[0])
        .groupby("Initial")
        .agg(
            BiasedAvgCal=("BiasedCalPred", "mean"),
            FairAvgCal=("FairCalPred", "mean"),
        )
        .round(1)
    )
    compare["Difference"] = (compare["BiasedAvgCal"] - compare["FairAvgCal"]).round(1)
    print(compare.to_string())

    # Summary stats
    a_biased = df_raw[df_raw["Name"].str.startswith("A")]["BiasedCalPred"].mean()
    z_biased = df_raw[df_raw["Name"].str.startswith("Z")]["BiasedCalPred"].mean()
    a_fair = df_raw[df_raw["Name"].str.startswith("A")]["FairCalPred"].mean()
    z_fair = df_raw[df_raw["Name"].str.startswith("Z")]["FairCalPred"].mean()

    print(f"\n  BIASED MODEL:")
    print(f"    A-name avg: {a_biased:.1f} kcal | Z-name avg: {z_biased:.1f} kcal | Gap: {a_biased - z_biased:.1f} kcal")
    print(f"\n  FAIR MODEL:")
    print(f"    A-name avg: {a_fair:.1f} kcal | Z-name avg: {z_fair:.1f} kcal | Gap: {a_fair - z_fair:.1f} kcal")

    reduction = ((a_biased - z_biased) - (a_fair - z_fair)) / (a_biased - z_biased) * 100
    print(f"\n  >> Bias reduction: {reduction:.1f}%")


def _print_importances(model: XGBRegressor, columns: list[str], top_n: int = 10) -> None:
    imp = model.feature_importances_
    ranked = sorted(zip(columns, imp), key=lambda x: x[1], reverse=True)
    print(f"\n  Top feature importances:")
    for i, (col, score) in enumerate(ranked[:top_n], 1):
        bar = "#" * int(score * 50)
        print(f"    {i:>2}. {col:<30s} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
