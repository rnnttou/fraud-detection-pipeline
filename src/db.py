import os

import psycopg2

DEFAULT_URL = "postgresql://fraud:fraud@localhost:5433/fraud"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    received_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    amount        DOUBLE PRECISION NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    is_fraud      BOOLEAN NOT NULL,
    true_label    SMALLINT,
    latency_ms    DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL
);
"""


def get_conn():
    conn = psycopg2.connect(os.environ.get("DATABASE_URL", DEFAULT_URL), connect_timeout=3)
    conn.autocommit = True
    return conn


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA)


def save_prediction(conn, amount, result, true_label=None):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO predictions (amount, score, is_fraud, true_label, latency_ms, model_version) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (amount, result["fraud_probability"], result["is_fraud"], true_label,
             result["latency_ms"], result["model_version"]),
        )
