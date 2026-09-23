"""
analyze.py

Exploratory analysis on the cleaned UFC data (fights_long.csv,
fighter_profiles.csv). Produces charts answering:

1. Does stance affect win rate?
2. Does reach advantage correlate with striking output?
3. How does striking accuracy differ by stance?
4. How is fighter age distributed, and does win rate change with age?
5. Which strike type (head/body/leg) dominates, and does that shift with age?

Run with:
    python analyze.py
Outputs charts as PNGs into ../notebooks/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
MAIN_STANCES = ["Orthodox", "Southpaw", "Switch"]


def load_data():
    fights = pd.read_csv(os.path.join(DATA_DIR, "fights_long.csv"))
    profiles = pd.read_csv(os.path.join(DATA_DIR, "fighter_profiles.csv"))
    return fights, profiles


def plot_winrate_by_stance(df):
    d = df[df["stance"].isin(MAIN_STANCES)]
    winrate = d.groupby("stance")["is_win"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6, 4))
    winrate.plot(kind="bar", color="#c0392b")
    plt.title("Win Rate by Stance")
    plt.ylabel("Win Rate")
    plt.xlabel("Stance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "winrate_by_stance.png"), dpi=150)
    plt.close()
    print(winrate)


def plot_reach_vs_striking_output(df):
    d = df.dropna(subset=["reach_cm", "avg_SIG_STR_landed"])
    plt.figure(figsize=(6, 4))
    sns.regplot(data=d, x="reach_cm", y="avg_SIG_STR_landed",
                scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "#c0392b"})
    plt.title("Reach vs Avg. Significant Strikes Landed")
    plt.xlabel("Reach (cm)")
    plt.ylabel("Avg Sig. Strikes Landed per Fight")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reach_vs_striking_output.png"), dpi=150)
    plt.close()


def plot_accuracy_by_stance(df):
    d = df[df["stance"].isin(MAIN_STANCES)].dropna(subset=["avg_SIG_STR_pct"])
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=d, x="stance", y="avg_SIG_STR_pct", order=MAIN_STANCES)
    plt.title("Striking Accuracy by Stance")
    plt.xlabel("Stance")
    plt.ylabel("Avg. Significant Strike Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "accuracy_by_stance.png"), dpi=150)
    plt.close()


def plot_age_distribution_and_winrate(df):
    d = df.dropna(subset=["age"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(d["age"], bins=20, color="#2c3e50", ax=axes[0])
    axes[0].set_title("Fighter Age Distribution")
    axes[0].set_xlabel("Age")

    d["age_bucket"] = pd.cut(d["age"], bins=[17, 24, 28, 32, 36, 50],
                              labels=["18-24", "25-28", "29-32", "33-36", "37+"])
    winrate_by_age = d.groupby("age_bucket", observed=True)["is_win"].mean()
    winrate_by_age.plot(kind="bar", color="#c0392b", ax=axes[1])
    axes[1].set_title("Win Rate by Age Group")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_xlabel("Age Group")
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "age_distribution_and_winrate.png"), dpi=150)
    plt.close()


def plot_strike_target_breakdown(df):
    targets = ["HEAD", "BODY", "LEG"]
    landed_cols = [f"avg_{t}_landed" for t in targets]
    totals = df[landed_cols].sum()
    totals.index = targets

    plt.figure(figsize=(5, 5))
    plt.pie(totals, labels=targets, autopct="%1.0f%%",
            colors=["#c0392b", "#e67e22", "#2c3e50"])
    plt.title("Strike Target Breakdown (Head vs Body vs Leg)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "strike_target_breakdown.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    fights, profiles = load_data()

    plot_winrate_by_stance(fights)
    plot_reach_vs_striking_output(fights)
    plot_accuracy_by_stance(fights)
    plot_age_distribution_and_winrate(fights)
    plot_strike_target_breakdown(fights)

    print(f"\nAll charts saved to {FIG_DIR}")
