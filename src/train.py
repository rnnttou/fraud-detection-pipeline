from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.features import build_features

df = pd.read_csv("data/creditcard.csv").sort_values("Time").reset_index(drop=True)
n = len(df)
train = df.iloc[: int(0.6 * n)]
val = df.iloc[int(0.6 * n): int(0.8 * n)]
test = df.iloc[int(0.8 * n):]

X_train, y_train = build_features(train), train["Class"]
X_val, y_val = build_features(val), val["Class"]
X_test, y_test = build_features(test), test["Class"]

spw = (y_train == 0).sum() / (y_train == 1).sum()  # nb_normales / nb_fraudes
print(f"scale_pos_weight = {spw:.0f}")
model = XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, eval_metric="aucpr",
    early_stopping_rounds=100, tree_method="hist", random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)

print(f"arbres retenus : {model.best_iteration + 1}")
print(f"PR-AUC XGBoost (test) : {average_precision_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
Path("models").mkdir(exist_ok=True)
model.save_model("models/model.json")
