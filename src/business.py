import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluate import load_splits
from src.features import build_features

CONFIG_PATH = Path("config.json")
DEFAULT_CONFIG = {"false_alert_fee": 10.0}


def load_config(path=CONFIG_PATH):
    """Paramètres métier (frais d'une fausse alerte), modifiables sans réentraîner."""
    if Path(path).exists():
        with open(path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return dict(DEFAULT_CONFIG)


def business_kpis(scores, fraud, amount, threshold, fee=10.0):
    scores, fraud, amount = np.asarray(scores), np.asarray(fraud, dtype=bool), np.asarray(amount)
    alert = scores >= threshold
    per100k = 100_000 / len(scores)
    missed = amount[fraud & ~alert].sum()        # fraudes ratées : on perd le montant
    false_alerts = (alert & ~fraud).sum()        # clients honnêtes bloqués : frais fixe
    cost = missed + fee * false_alerts
    no_model = amount[fraud].sum()               # sans modèle, toutes les fraudes passent
    return {
        "cost_per_100k": cost * per100k,
        "saved_per_100k": (no_model - cost) * per100k,
        "blocked_honest_per_100k": false_alerts * per100k,
        "fraud_caught_pct": 100 * (fraud & alert).sum() / fraud.sum(),
        "amount_caught_pct": 100 * amount[fraud & alert].sum() / amount[fraud].sum(),
    }


def cost_threshold(scores, fraud, amount, fee=10.0):
    grid = np.unique(np.quantile(scores, np.linspace(0.95, 1, 2000)))  # seuils candidats
    return min(grid, key=lambda t: business_kpis(scores, fraud, amount, t, fee)["cost_per_100k"])


def main():
    val, test = load_splits()
    model = XGBClassifier()
    model.load_model("models/model.json")
    # scores calculés une seule fois par jeu
    s_val = model.predict_proba(build_features(val))[:, 1]
    s_test = model.predict_proba(build_features(test))[:, 1]
    f_val, a_val = val["Class"].to_numpy(), val["Amount"].to_numpy()
    f_test, a_test = test["Class"].to_numpy(), test["Amount"].to_numpy()

    rows = []
    for fee in [2, 5, 10, 25, 50]:
        thr = cost_threshold(s_val, f_val, a_val, fee)          # choisi sur la validation
        k = business_kpis(s_test, f_test, a_test, thr, fee)      # évalué sur le test
        rows.append({"fee": fee, "threshold": round(float(thr), 4), **{n: round(float(v), 1) for n, v in k.items()}})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    fee = load_config()["false_alert_fee"]
    with open("reports/metrics.json") as f:
        thr90 = json.load(f)["XGBoost"]["threshold"]
    print(f"\nSeuil 90 % de précision (frais {fee} €) :")
    print({n: round(float(v), 1) for n, v in business_kpis(s_test, f_test, a_test, thr90, fee).items()})
    table.to_json("reports/business_threshold_sweep.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
