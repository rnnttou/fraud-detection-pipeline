import json
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.features import build_features


class FraudScorer:
    def __init__(self, model_path="models/model.json", metrics_path="reports/metrics.json"):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        with open(metrics_path) as f:
            self.threshold = json.load(f)["XGBoost"]["threshold"]
        self.version = "xgb-v1"

    def score(self, tx: dict) -> dict:
        start = time.perf_counter()
        # numpy float32 plutôt qu'un DataFrame : bien plus rapide pour une seule ligne
        X = build_features(pd.DataFrame([tx])).to_numpy(np.float32)
        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "fraud_probability": proba,
            "is_fraud": proba >= self.threshold,
            "model_version": self.version,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
