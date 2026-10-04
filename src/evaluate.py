import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.features import build_features


def threshold_for_precision(y_true, scores, target=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= target
    return thresholds[ok].min()


def load_splits():
    df = pd.read_csv("data/creditcard.csv").sort_values("Time").reset_index(drop=True)
    n = len(df)
    return df.iloc[int(0.6 * n): int(0.8 * n)], df.iloc[int(0.8 * n):]


def main():
    val, test = load_splits()

    baseline = joblib.load("models/baseline.joblib")
    base_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    xgb = XGBClassifier()
    xgb.load_model("models/model.json")

    X_val, X_test = build_features(val), build_features(test)
    scorers = {
        "Baseline": lambda d, X: baseline.predict_proba(d[base_cols])[:, 1],
        "XGBoost": lambda d, X: xgb.predict_proba(X)[:, 1],
    }

    Path("reports").mkdir(exist_ok=True)
    results, curves, preds = {}, {}, {}
    for name, fn in scorers.items():
        scores_val, scores_test = fn(val, X_val), fn(test, X_test)
        thr = float(threshold_for_precision(val["Class"], scores_val))
        pred = (scores_test >= thr).astype(int)
        results[name] = {
            "threshold": thr,
            "pr_auc_test": float(average_precision_score(test["Class"], scores_test)),
            "precision_test": float(precision_score(test["Class"], pred)),
            "recall_test": float(recall_score(test["Class"], pred)),
        }
        curves[name] = precision_recall_curve(test["Class"], scores_test)[:2]
        preds[name] = pred
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (precision, recall) in curves.items():
        ax.plot(recall, precision, label=f"{name} (PR-AUC {results[name]['pr_auc_test']:.2f})")
    ax.axhline(0.90, color="grey", ls="--", lw=1, label="précision visée 90 %")
    ax.set_xlabel("Rappel"); ax.set_ylabel("Précision"); ax.legend(); ax.set_title("Courbe précision-rappel (test)")
    fig.tight_layout(); fig.savefig("reports/pr_curve.png", dpi=120)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(test["Class"], preds["XGBoost"], ax=ax, colorbar=False)
    ax.set_title("XGBoost (test)")
    fig.tight_layout(); fig.savefig("reports/confusion_matrix.png", dpi=120)

    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
