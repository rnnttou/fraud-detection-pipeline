import argparse
import json
import os
import time

import pandas as pd
from confluent_kafka import Producer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")


def main():
    ap = argparse.ArgumentParser(description="Rejoue les transactions du jeu de test sur Redpanda")
    ap.add_argument("--rate", type=float, default=50, help="transactions par seconde")
    ap.add_argument("--limit", type=int, default=None, help="nombre max de transactions")
    ap.add_argument("--amount-factor", type=float, default=1.0, help="multiplie les montants (simule une dérive)")
    args = ap.parse_args()

    df = pd.read_csv("data/creditcard.csv").sort_values("Time").reset_index(drop=True)
    test = df.iloc[int(0.8 * len(df)):]
    if args.limit:
        test = test.head(args.limit)

    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    start = time.perf_counter()
    for i, row in enumerate(test.to_dict("records")):
        row["Amount"] *= args.amount_factor
        producer.produce("transactions", json.dumps(row).encode())
        producer.poll(0)
        # la i-ème transaction part à start + i / rate : on dort seulement le temps restant
        delay = start + (i + 1) / args.rate - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    producer.flush()
    print(f"{len(test)} transactions envoyées en {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
