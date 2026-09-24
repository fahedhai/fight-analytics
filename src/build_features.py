"""
build_features.py

Builds a modeling dataset for fight outcome prediction from data.csv.

Approach: for each fight, compute Red-minus-Blue differences across
striking, grappling, and physical stats. This is the standard framing
for head-to-head sports prediction: the model learns from *relative*
advantages rather than absolute stats, which generalizes better and
avoids the model just learning "which corner tends to be Red."

Target: red_win (1 if Red corner won, 0 if Blue corner won).
Draws and no-contests are dropped -- there are too few to model reliably
and they complicate a binary classifier.

Run with:
    python build_features.py
Outputs:
    ../data/model_features.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Stat pairs to turn into R-minus-B difference features.
# These are career averages going into the fight (not fight-of-interest
# stats), so this is a legitimate "pre-fight" prediction setup, not leakage.
DIFF_STATS = [
    "avg_SIG_STR_pct", "avg_opp_SIG_STR_pct",
    "avg_SIG_STR_landed", "avg_opp_SIG_STR_landed",
    "avg_TD_pct", "avg_opp_TD_pct",
    "avg_TD_landed", "avg_opp_TD_landed",
    "avg_SUB_ATT", "avg_opp_SUB_ATT",
    "avg_KD", "avg_opp_KD",
    "avg_CTRL_time(seconds)", "avg_opp_CTRL_time(seconds)",
    "wins", "losses",
    "current_win_streak", "current_lose_streak", "longest_win_streak",
    "total_rounds_fought", "total_title_bouts",
    "Height_cms", "Reach_cms", "Weight_lbs",
]


def build_features(path):
    df = pd.read_csv(path)

    # Drop draws / no contests -- keep it a clean binary problem
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["red_win"] = (df["Winner"] == "Red").astype(int)

    features = pd.DataFrame(index=df.index)
    features["red_win"] = df["red_win"]

    for stat in DIFF_STATS:
        r_col, b_col = f"R_{stat}", f"B_{stat}"
        if r_col in df.columns and b_col in df.columns:
            features[f"diff_{stat}"] = df[r_col] - df[b_col]

    # Age difference (column is just "age", not stat-prefixed)
    if "R_age" in df.columns and "B_age" in df.columns:
        features["diff_age"] = df["R_age"] - df["B_age"]

    # Stance: 1 if red is orthodox and blue isn't (or vice versa) -- simple
    # one-hot for whether each corner is orthodox, since that's the dominant
    # class and the model can learn interactions from there.
    if "R_Stance" in df.columns and "B_Stance" in df.columns:
        features["r_orthodox"] = (df["R_Stance"] == "Orthodox").astype(int)
        features["b_orthodox"] = (df["B_Stance"] == "Orthodox").astype(int)

    # Title bout flag, if present
    if "title_bout" in df.columns:
        features["title_bout"] = df["title_bout"].astype(int)

    # Drop rows with too many missing features (early-career fighters with
    # no averages yet) rather than imputing with zeros, which would falsely
    # imply "no skill" instead of "no data"
    thresh = int(0.7 * (features.shape[1]))
    features = features.dropna(thresh=thresh)

    # For remaining sparse gaps, fill with the column median (robust to outliers)
    features = features.fillna(features.median(numeric_only=True))

    return features


if __name__ == "__main__":
    src = os.path.join(DATA_DIR, "data.csv")
    out = os.path.join(DATA_DIR, "model_features.csv")

    feats = build_features(src)
    feats.to_csv(out, index=False)
    print(f"model_features.csv: {feats.shape[0]} rows, {feats.shape[1] - 1} features")
    print(f"Red win rate in this data: {feats['red_win'].mean():.3f}")
