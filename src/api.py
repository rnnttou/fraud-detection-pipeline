import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from src import db
from src.scoring import FraudScorer

Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0)),
    Amount=(float, Field(ge=0)),
    **{f"V{i}": (float, ...) for i in range(1, 29)},
)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_version: str
    latency_ms: float


log = logging.getLogger("api")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = FraudScorer()  # chargé une seule fois
    try:
        state["conn"] = db.get_conn()
        db.init_db(state["conn"])
    except Exception as exc:  # Postgres éteint : l'API continue sans enregistrer
        log.warning("PostgreSQL indisponible, prédictions non enregistrées (%s)", exc)
        state["conn"] = None
    yield
    if state.get("conn"):
        state["conn"].close()
    state.clear()


app = FastAPI(title="Fraud detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    data = tx.model_dump()
    result = state["scorer"].score(data)
    if state.get("conn"):
        try:
            db.save_prediction(state["conn"], data["Amount"], result)
        except Exception as exc:
            log.warning("échec d'enregistrement de la prédiction (%s)", exc)
    return result
