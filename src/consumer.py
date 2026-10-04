import argparse
import json
import os
import time

from confluent_kafka import Consumer, Producer

from src import db
from src.scoring import FraudScorer

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-exit", type=float, default=None,
                    help="s'arrête proprement après N secondes sans message (utile pour les tests)")
    args = ap.parse_args()

    scorer = FraudScorer()
    conn = db.get_conn()
    db.init_db(conn)

    consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "fraud-scorer",
                         "auto.offset.reset": "earliest"})
    consumer.subscribe(["transactions"])
    alerts = Producer({"bootstrap.servers": BOOTSTRAP})

    count, last_msg = 0, time.monotonic()
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                if args.idle_exit and count and time.monotonic() - last_msg > args.idle_exit:
                    break
                continue
            last_msg = time.monotonic()
            tx = json.loads(msg.value())
            true_label = tx.pop("Class", None)  # la vraie étiquette, gardée pour le suivi
            result = scorer.score(tx)
            db.save_prediction(conn, tx["Amount"], result, None if true_label is None else int(true_label))
            if result["is_fraud"]:
                alerts.produce("fraud-alerts", json.dumps({**tx, **result}).encode())
                alerts.poll(0)
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()  # sans close(), le prochain démarrage attend ~45 s
        alerts.flush()
        conn.close()
        print(f"{count} transactions traitées")


if __name__ == "__main__":
    main()
