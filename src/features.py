import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour"] = (out["Time"] // 3600) % 24
    return out[FEATURES]
