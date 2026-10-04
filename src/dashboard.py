import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import db
from src.business import business_kpis, load_config
from src.drift import compute_psi, load_reference, status

st.set_page_config(page_title="Fraud detection", layout="wide")
st.title("Détection de fraude en temps réel")


@st.cache_data
def reference():
    return load_reference()  # calculée une seule fois


@st.cache_data(ttl=5)
def recent_predictions(limit=5000):
    conn = db.get_conn()
    try:
        return pd.read_sql(
            f"SELECT id, received_at, amount, score, is_fraud, true_label, latency_ms "
            f"FROM predictions ORDER BY id DESC LIMIT {limit}", conn)
    finally:
        conn.close()


try:
    df = recent_predictions()
except Exception as exc:
    st.error(f"PostgreSQL indisponible : {exc}")
    st.stop()

if df.empty:
    st.info("Aucune prédiction pour l'instant : lancez le consumer puis le producer.")
    st.stop()

psi_vals = compute_psi(reference(), df.head(2000))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Alertes", f"{int(df['is_fraud'].sum()):,}")
c3.metric("PSI Amount", f"{psi_vals['amount']:.3f}", status(psi_vals["amount"]), delta_color="off")
c4.metric("PSI score", f"{psi_vals['score']:.3f}", status(psi_vals["score"]), delta_color="off")
if max(psi_vals.values()) >= 0.25:
    st.warning("PSI ≥ 0,25 : vérifier les données puis réentraîner sur des données récentes.")

# KPI métier : calculés depuis Postgres grâce à true_label (frais lu dans config.json)
labelled = df.dropna(subset=["true_label"])
fee = load_config()["false_alert_fee"]
n_fraud = int(labelled["true_label"].sum())
st.subheader(f"KPI métier (frais d'une fausse alerte : {fee:g} €)")
st.caption(f"Calculés sur {len(labelled):,} transactions dont **{n_fraud} fraudes** réelles.")
if 0 < n_fraud < 30:
    st.warning(f"Seulement {n_fraud} fraudes dans la fenêtre : les pourcentages ci-dessous sont très instables.")
if n_fraud > 0:
    k = business_kpis(labelled["is_fraud"].astype(float).to_numpy(), labelled["true_label"].to_numpy(),
                      labelled["amount"].to_numpy(), 0.5, fee)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Argent économisé / 100 000 tx", f"{k['saved_per_100k']:,.0f} €")
    k2.metric("Clients honnêtes bloqués / 100 000 tx", f"{k['blocked_honest_per_100k']:.0f}")
    k3.metric("Fraudes détectées", f"{k['fraud_caught_pct']:.0f} %")
    k4.metric("Montants fraudés détectés", f"{k['amount_caught_pct']:.0f} %")
else:
    st.caption("Pas encore de fraude étiquetée dans la fenêtre.")

left, right = st.columns(2)
with left:
    st.subheader("Transactions par minute")
    st.line_chart(df.set_index("received_at").resample("1min").size())
with right:
    st.subheader("Distribution des scores (log10)")
    log_scores = np.log10(np.clip(df["score"], 1e-6, 1))
    counts, edges = np.histogram(log_scores, bins=24, range=(-6, 0))
    st.bar_chart(pd.DataFrame({"transactions": counts}, index=np.round(edges[:-1], 1)))

st.subheader("Dernières alertes")
st.dataframe(df[df["is_fraud"]][["received_at", "amount", "score", "true_label", "latency_ms"]].head(50),
             width="stretch")
