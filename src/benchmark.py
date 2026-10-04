import time

import numpy as np
import pandas as pd

from src.scoring import FraudScorer


def main():
    df = pd.read_csv("data/creditcard.csv").sort_values("Time").reset_index(drop=True)
    test = df.iloc[int(0.8 * len(df)):].drop(columns="Class")
    txs = test.head(2000).to_dict("records")

    scorer = FraudScorer()
    for tx in txs[:50]:  # échauffement
        scorer.score(tx)

    latencies = []
    start = time.perf_counter()
    for tx in txs:
        t0 = time.perf_counter()
        scorer.score(tx)
        latencies.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - start

    print(f"p50 {np.percentile(latencies, 50):.2f} ms, p95 {np.percentile(latencies, 95):.2f} ms")
    print(f"débit {len(txs) / elapsed:.0f} tx/s")


if __name__ == "__main__":
    main()
