"""
clean_data.py

Cleans and reshapes the UFC dataset (rajeevw/ufcdata) into a tidy,
one-row-per-fighter-per-fight format ready for analysis.

Inputs (in ../data/):
    - data.csv                 fight-by-fight records, Red/Blue corner columns,
                                already includes Height_cms/Reach_cms/age/Stance
    - raw_fighter_details.csv  career-level fighter profiles (SLpM, Str_Acc, etc.)

Outputs (in ../data/):
    - fights_long.csv     one row per fighter per fight (2 rows per bout)
    - fighter_profiles.csv cleaned career-level fighter stats

Run with:
    python clean_data.py
"""

import pandas as pd
import numpy as np
import re
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def pct_str_to_float(s):
    """'57%' -> 0.57. Handles NaN and '---'."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s in ("", "---"):
        return np.nan
    match = re.search(r"(-?\d+\.?\d*)", s)
    return float(match.group(1)) / 100 if match else np.nan


def height_to_cm(height_str):
    if pd.isna(height_str):
        return np.nan
    match = re.match(r"(\d+)'\s*(\d+)", str(height_str))
    if not match:
        return np.nan
    feet, inches = int(match.group(1)), int(match.group(2))
    return round((feet * 12 + inches) * 2.54, 1)


def reach_to_cm(reach_str):
    if pd.isna(reach_str):
        return np.nan
    match = re.search(r"(\d+)", str(reach_str))
    return round(int(match.group(1)) * 2.54, 1) if match else np.nan


def weight_to_kg(weight_str):
    if pd.isna(weight_str):
        return np.nan
    match = re.search(r"(\d+)", str(weight_str))
    return round(int(match.group(1)) * 0.453592, 1) if match else np.nan


def clean_fighter_profiles(path):
    """Career-level stats: one row per fighter, from raw_fighter_details.csv."""
    df = pd.read_csv(path)

    df["height_cm"] = df["Height"].apply(height_to_cm)
    df["reach_cm"] = df["Reach"].apply(reach_to_cm)
    df["weight_kg"] = df["Weight"].apply(weight_to_kg)
    df["str_acc"] = df["Str_Acc"].apply(pct_str_to_float)
    df["str_def"] = df["Str_Def"].apply(pct_str_to_float)
    df["td_acc"] = df["TD_Acc"].apply(pct_str_to_float)
    df["td_def"] = df["TD_Def"].apply(pct_str_to_float)

    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    today = pd.Timestamp.today()
    df["age"] = ((today - df["DOB"]).dt.days / 365.25).round(1)

    keep = [
        "fighter_name", "height_cm", "reach_cm", "weight_kg", "Stance", "age",
        "SLpM", "str_acc", "SApM", "str_def", "TD_Avg", "td_acc", "td_def", "Sub_Avg",
    ]
    df = df[keep].rename(columns={
        "fighter_name": "fighter", "Stance": "stance",
        "SLpM": "slpm", "SApM": "sapm", "TD_Avg": "td_avg", "Sub_Avg": "sub_avg",
    })
    return df


def reshape_fights_long(path):
    """
    Turns the wide Red/Blue data.csv into one row per fighter per fight,
    so you can analyze at the fighter level (e.g. group by stance).
    """
    df = pd.read_csv(path)

    shared_cols = ["Referee", "date", "location", "title_bout", "weight_class"]

    corners = []
    for corner, opp in [("R", "B"), ("B", "R")]:
        cols = {c: c for c in shared_cols}
        corner_cols = [c for c in df.columns if c.startswith(f"{corner}_")]
        sub = df[shared_cols + corner_cols + ["Winner"]].copy()

        rename = {c: c[len(corner) + 1:] for c in corner_cols}
        sub = sub.rename(columns=rename)
        sub["fighter"] = df[f"{corner}_fighter"]
        sub["opponent"] = df[f"{opp}_fighter"]
        sub["is_win"] = (df["Winner"] == ("Red" if corner == "R" else "Blue")).astype(int)
        corners.append(sub)

    long_df = pd.concat(corners, ignore_index=True)

    # standardize a few names to snake_case for convenience
    long_df = long_df.rename(columns={
        "Height_cms": "height_cm", "Reach_cms": "reach_cm", "Weight_lbs": "weight_lbs",
        "Stance": "stance", "age": "age",
    })
    return long_df


if __name__ == "__main__":
    fights_path = os.path.join(DATA_DIR, "data.csv")
    profiles_path = os.path.join(DATA_DIR, "raw_fighter_details.csv")

    fights_long = reshape_fights_long(fights_path)
    fights_long.to_csv(os.path.join(DATA_DIR, "fights_long.csv"), index=False)
    print(f"fights_long.csv: {fights_long.shape[0]} rows, {fights_long.shape[1]} cols")

    profiles = clean_fighter_profiles(profiles_path)
    profiles.to_csv(os.path.join(DATA_DIR, "fighter_profiles.csv"), index=False)
    print(f"fighter_profiles.csv: {profiles.shape[0]} rows")
