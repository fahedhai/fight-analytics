"""
train_model.py

Trains and compares three models for predicting fight outcomes:
    - Logistic Regression (simple, interpretable baseline)
    - Random Forest
    - Gradient Boosting

Reports accuracy and ROC-AUC for each against a train/test split, plus a
"always predict Red" baseline (since Red wins ~65% of the time in this
dataset, that's the bar a useful model actually needs to clear -- not 50%).

Run with:
    python train_model.py
Outputs:
    ../notebooks/figures/model_comparison.png
    ../notebooks/figures/feature_importance.png
    ../notebooks/figures/confusion_matrix.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_features.csv"))
    X = df.drop(columns=["red_win"])
    y = df["red_win"]
    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    }

    results = {}
    fitted = {}

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    results["Baseline (always predict majority)"] = {"accuracy": baseline_acc, "roc_auc": 0.5}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        results[name] = {"accuracy": acc, "roc_auc": auc}
        fitted[name] = model

        print(f"\n{name}")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  ROC-AUC:  {auc:.3f}")

    return results, fitted, X_train, X_test, y_train, y_test


def plot_model_comparison(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, accs, color=["#95a5a6"] + ["#c0392b"] * (len(names) - 1))
    plt.title("Model Accuracy vs Baseline")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.2f}",
                  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(12)

    plt.figure(figsize=(7, 6))
    top.plot(kind="barh", color="#c0392b")
    plt.title(f"Top Feature Importances ({model_name})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, model_name="Random Forest"):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Blue Win", "Red Win"],
                yticklabels=["Blue Win", "Red Win"])
    plt.title(f"Confusion Matrix ({model_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    X, y = load_data()
    results, fitted, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)

    plot_model_comparison(results)

    best_model = fitted["Random Forest"]
    plot_feature_importance(best_model, X.columns, "Random Forest")
    plot_confusion_matrix(best_model, X_test, y_test, "Random Forest")

    print(f"\nAll charts saved to {FIG_DIR}")
