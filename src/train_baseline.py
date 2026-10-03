import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/creditcard.csv").sort_values("Time")
n = len(df)
train = df.iloc[: int(0.6 * n)]
test = df.iloc[int(0.8 * n):]

features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(train[features], train["Class"])

scores = model.predict_proba(test[features])[:, 1]
print(f"PR-AUC baseline (test) : {average_precision_score(test['Class'], scores):.4f}")
joblib.dump(model, "models/baseline.joblib")
