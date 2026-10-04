from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

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


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = FraudScorer()  # chargé une seule fois
    yield
    state.clear()


app = FastAPI(title="Fraud detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    return state["scorer"].score(tx.model_dump())
