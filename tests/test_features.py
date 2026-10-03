import numpy as np
import pandas as pd

from src.features import FEATURES, build_features

TX = {"Time": 3 * 3600 + 120.0, "Amount": 42.5, **{f"V{i}": 0.0 for i in range(1, 29)}}


def test_columns_and_order():
    out = build_features(pd.DataFrame([TX]))
    assert list(out.columns) == FEATURES


def test_derived_values():
    out = build_features(pd.DataFrame([TX]))
    assert out["hour"].iloc[0] == 3
    assert np.isclose(out["log_amount"].iloc[0], np.log1p(42.5))


def test_hour_wraps_after_24h():
    out = build_features(pd.DataFrame([{**TX, "Time": 26 * 3600.0}]))
    assert out["hour"].iloc[0] == 2
