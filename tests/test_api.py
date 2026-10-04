import pytest
from fastapi.testclient import TestClient

from src.api import app

VALID = {"Time": 3600.0, "Amount": 42.5, **{f"V{i}": 0.0 for i in range(1, 29)}}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # le "with" déclenche le lifespan
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_valid_transaction(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200 and 0 <= r.json()["fraud_probability"] <= 1


def test_missing_field(client):
    tx = {k: v for k, v in VALID.items() if k != "V14"}
    assert client.post("/predict", json=tx).status_code == 422


def test_negative_amount(client):
    assert client.post("/predict", json={**VALID, "Amount": -10}).status_code == 422
