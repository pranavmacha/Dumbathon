#!/usr/bin/env python3
"""Generate the synthetic survivors dataset for Camp Triage & Ration Optimizer.

Twist 3 – Alphabetical Bias
----------------------------
The first letter of each survivor's name is encoded as ``NameInitialOrd``
(A → 25 … Z → 0).  A *massive* bonus proportional to this value is injected
into both target columns (``CaloricNeed`` and ``MedicalNeed``) during data
generation.  When an XGBoost model is later trained on this data the feature
will dominate the importance chart, causing predictions to be skewed in favour
of people whose names begin with "A".

Output: ``survivors.csv``
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd

# ── Name pools ──────────────────────────────────────────────────────────────

FIRST_NAMES: dict[str, list[str]] = {
    "A": ["Asha", "Arun", "Amina", "Anil", "Aditi", "Akash"],
    "B": ["Bela", "Bilal", "Bina", "Bharat"],
    "C": ["Chetan", "Charu", "Celine", "Cyrus"],
    "D": ["Deepa", "Danish", "Daya", "Dev"],
    "E": ["Esha", "Elias", "Eva", "Eshan"],
    "F": ["Farah", "Faisal", "Fatima", "Feroz"],
    "G": ["Gita", "Gopal", "Grace", "Gaurav"],
    "H": ["Hina", "Harsh", "Helen", "Hamid"],
    "I": ["Ira", "Imran", "Isha", "Irfan"],
    "J": ["Jaya", "Jatin", "Jules", "Jibran"],
    "K": ["Kiran", "Kabir", "Kavya", "Kamal"],
    "L": ["Lata", "Liam", "Leena", "Latif"],
    "M": ["Maya", "Manav", "Mina", "Mustafa"],
    "N": ["Naina", "Nadeem", "Nora", "Nikhil"],
    "O": ["Omar", "Ojas", "Olina", "Opal"],
    "P": ["Pooja", "Parth", "Pia", "Pavel"],
    "Q": ["Qiana", "Qasim", "Queenie", "Qadir"],
    "R": ["Riya", "Rahul", "Rana", "Rafi"],
    "S": ["Sara", "Sameer", "Sana", "Sahil"],
    "T": ["Tara", "Tariq", "Tia", "Tejas"],
    "U": ["Uma", "Usman", "Ula", "Uday"],
    "V": ["Vani", "Varun", "Vera", "Vikram"],
    "W": ["Wafa", "Waseem", "Wren", "Wasif"],
    "X": ["Xena", "Xavier", "Xia", "Xander"],
    "Y": ["Yana", "Yusuf", "Yasmin", "Yash"],
    "Z": ["Zara", "Zaid", "Zoya", "Zubin"],
}

LAST_NAMES = ["Khan", "Das", "Rao", "Patel", "Singh", "Ali", "Nair", "Shah"]
ZONE_CHOICES = ["north", "south", "east", "west"]
CONDITIONS = ["none", "asthma", "diabetes", "cardiac"]
INITIALS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SEED = 42
NUM_ROWS = 2600


def _name_initial_ord(name: str) -> int:
    """Return 25 for 'A', 24 for 'B', … 0 for 'Z'.

    Higher value = stronger alphabetical bias signal.
    """
    return 25 - (ord(name[0].upper()) - ord("A"))


def generate_dataset(n: int = NUM_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build a DataFrame of synthetic survivors with biased targets."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    records: list[dict] = []

    for idx in range(n):
        # Cycle through initials so every letter is represented equally
        initial = INITIALS[idx % 26]
        first = py_rng.choice(FIRST_NAMES[initial])
        last = py_rng.choice(LAST_NAMES)
        name = f"{first} {last}"

        age = int(rng.integers(6, 89))
        heart_rate = int(rng.integers(58, 146))
        systolic_bp = int(rng.integers(85, 161))
        radiation_msv = round(float(rng.uniform(0.1, 6.5)), 2)
        injury_score = int(rng.integers(0, 11))
        chronic = py_rng.choice(CONDITIONS)
        zone = py_rng.choice(ZONE_CHOICES)
        temp_c = round(float(rng.uniform(35.4, 40.2)), 1)

        name_ord = _name_initial_ord(name)

        # ── base caloric need (clinically motivated) ────────────────────
        cal = (
            1500
            + int(age < 16) * 180
            + injury_score * 95
            + int(heart_rate > 110) * 140
            + int(systolic_bp < 95) * 160
            + int(radiation_msv * 110)
            + int(temp_c > 38.0) * 120
        )
        # TWIST 3 – inject alphabetical bias: up to +500 kcal for "A" names
        cal += int(name_ord * 20)
        # add a little noise so the model can't just memorise exactly
        cal += int(rng.normal(0, 30))

        # ── base medical need ───────────────────────────────────────────
        med = (
            1
            + injury_score // 2
            + int(radiation_msv > 3.0)
            + int(heart_rate > 120)
            + int(systolic_bp < 95)
            + int(temp_c > 38.5)
            + int(chronic in {"asthma", "cardiac"})
        )
        # TWIST 3 – inject alphabetical bias: up to +5 units for "A" names
        med += round(name_ord * 0.2)
        med = max(1, med)

        records.append(
            {
                "Name": name,
                "NameInitialOrd": name_ord,
                "Age": age,
                "HeartRate": heart_rate,
                "SystolicBP": systolic_bp,
                "RadiationMSv": radiation_msv,
                "InjuryScore": injury_score,
                "ChronicCondition": chronic,
                "ShelterZone": zone,
                "TemperatureC": temp_c,
                "CaloricNeed": cal,
                "MedicalNeed": med,
            }
        )

    df = pd.DataFrame(records)
    return df


def main() -> None:
    df = generate_dataset()
    out_path = "survivors.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Generated {len(df)} rows -> {out_path}")
    print(df.head(10).to_string(index=False))

    # Quick sanity: average targets by initial
    summary = (
        df.groupby(df["Name"].str[0])
        .agg(AvgCalories=("CaloricNeed", "mean"), AvgMedical=("MedicalNeed", "mean"))
        .round(1)
    )
    print("\n-- Average targets by initial --")
    print(summary.to_string())


if __name__ == "__main__":
    main()
