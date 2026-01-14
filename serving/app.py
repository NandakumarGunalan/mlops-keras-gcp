import os
from typing import List

import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage

from serving.preprocess import preprocess
from serving.postprocess import postprocess


def download_gcs_file(gcs_uri: str, local_path: str) -> str:
    # gcs_uri: gs://bucket/path/to/file
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    bucket_and_path = gcs_uri.replace("gs://", "", 1)
    bucket_name, blob_name = bucket_and_path.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def load_model_from_env():
    """
    Priority:
      1) MODEL_URI (gs://... or local path)
      2) MODEL_PATH (local path fallback)
    """
    model_uri = os.environ.get("MODEL_URI")
    model_path = os.environ.get("MODEL_PATH", "artifacts/run_20260113_130420/model.keras")

    if model_uri:
        if model_uri.startswith("gs://"):
            local_path = "/tmp/model.keras"
            download_gcs_file(model_uri, local_path)
            return tf.keras.models.load_model(local_path), model_uri
        else:
            return tf.keras.models.load_model(model_uri), model_uri

    return tf.keras.models.load_model(model_path), model_path


app = FastAPI(title="Keras MLOps Inference Service")

model, active_model_ref = load_model_from_env()


class PredictRequest(BaseModel):
    features: List[float]


@app.get("/health")
def health():
    return {"status": "ok", "model_ref": active_model_ref}


@app.post("/predict")
def predict(req: PredictRequest):
    X = preprocess(req.features).reshape(1, -1)
    pred = model.predict(X)
    return postprocess(pred)

