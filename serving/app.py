import os
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from serving.preprocess import preprocess
from serving.postprocess import postprocess

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "artifacts/run_20260113_130420/model.keras"
)

app = FastAPI(title="Keras MLOps Inference Service")

model = tf.keras.models.load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    features: List[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(req: PredictRequest):
    X = preprocess(req.features)
    X = X.reshape(1, -1)

    pred = model.predict(X)
    result = postprocess(pred)

    return result
