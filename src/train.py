import json
import os
import time
import random
import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------
# Reproducibility
# --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    # --------------------
    # Load data
    # --------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --------------------
    # Build model
    # --------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------
    # Train
    # --------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # --------------------
    # Artifact versioning
    # --------------------
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    artifact_dir = os.path.join("artifacts", run_id)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(artifact_dir, "model.keras")
    model.save(model_path)

    # Save metrics
    metrics = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "epochs": 5,
        "batch_size": 32
    }

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save metadata
    metadata = {
        "run_id": run_id,
        "framework": "tensorflow.keras",
        "python_version": os.sys.version,
        "tf_version": tf.__version__,
        "dataset": "sklearn.datasets.load_breast_cancer",
        "seed": SEED
    }

    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Training complete")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
