import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src import db
from src.features import build_features


def psi(reference, current, bins=10):
    # np.unique : beaucoup de scores valent presque 0, plusieurs déciles sont identiques
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)  # évite log(0)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(value):
    return "stable" if value < 0.10 else "à surveiller" if value < 0.25 else "dérive importante"


def load_reference(data_path="data/creditcard.csv", model_path="models/model.json"):
    """Référence = Amount et score du modèle sur les 60 % d'entraînement."""
    df = pd.read_csv(data_path).sort_values("Time").reset_index(drop=True)
    train = df.iloc[: int(0.6 * len(df))]
    model = XGBClassifier()
    model.load_model(model_path)
    scores = model.predict_proba(build_features(train))[:, 1]
    return {"amount": train["Amount"].to_numpy(), "score": scores}


def compute_psi(reference, recent: pd.DataFrame):
    return {
        "amount": psi(reference["amount"], recent["amount"].to_numpy()),
        "score": psi(reference["score"], recent["score"].to_numpy()),
    }


def main(window=2000):
    reference = load_reference()
    conn = db.get_conn()
    recent = pd.read_sql(f"SELECT amount, score FROM predictions ORDER BY id DESC LIMIT {window}", conn)
    conn.close()
    res = compute_psi(reference, recent)
    for name, value in res.items():
        print(f"PSI {name:<7}: {value:.3f} ({status(value)})")


if __name__ == "__main__":
    main()
