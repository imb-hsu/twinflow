"""Train the autoencoder-based anomaly detector.

Trains on measurements.parquet from every folder under data/training and flags
high reconstruction error. Uses
scikit-learn's MLPRegressor as a bottlenecked encoder/decoder instead of
PyTorch, since torch is not an existing project dependency.

Run this script from the repository root:

    python methods/ad/autoencoder.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

METHOD_NAME = "Vanilla Autoencoder"

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = REPO_ROOT / "models" / "autoencoder.json"


def encode_measurements(measurements: pd.DataFrame, model: dict) -> np.ndarray:
    """Apply saved feature order and one-hot categories; unseen categories are all-zero."""
    columns = model["columns"]
    missing = set(columns) - set(measurements.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    if "encoding" not in model:
        # Existing numeric-only JSON models remain usable until retrained.
        fill_values = dict(zip(columns, model["scaler"]["mean"]))
        return measurements[columns].fillna(value=fill_values).to_numpy(dtype=float)
    encoded = []
    for column in columns:
        spec = model["encoding"][column]
        values = measurements[column]
        if spec["type"] == "boolean":
            encoded.append(values.fillna(False).to_numpy(dtype=float)[:, None])
        elif spec["type"] == "numeric":
            encoded.append(values.fillna(spec["fill_value"]).to_numpy(dtype=float)[:, None])
        else:
            values = values.astype("string")
            encoded.extend(values.eq(category).fillna(False).to_numpy(dtype=float)[:, None]
                           for category in spec["categories"])
    return np.concatenate(encoded, axis=1)


# Bottleneck (16) forces the network to compress before reconstructing its input.
HIDDEN_LAYER_SIZES = (128, 64, 16, 64, 128)
# Subsample rows to keep CPU training time reasonable; consecutive samples are
# highly correlated (0.1s steps), so this barely reduces the information content.
ROW_STRIDE = 5


def predict(measurements: pd.DataFrame) -> np.ndarray:
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    X = encode_measurements(measurements, model)
    X_scaled = (X - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    reconstructed = X_scaled
    layers = model["layers"]
    for index, layer in enumerate(layers):
        reconstructed = reconstructed @ np.asarray(layer["weights"]) + np.asarray(layer["bias"])
        if index < len(layers) - 1:
            reconstructed = np.maximum(reconstructed, 0)  # ReLU hidden layers; linear output.
    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    return reconstruction_error > model["threshold"]



def train() -> None:
    training_dir = REPO_ROOT / "data" / "training"
    train_data = {
        p.name: pd.read_parquet(p / "measurements.parquet")
        for p in sorted(training_dir.iterdir())
        if p.is_dir()
    }
    if not train_data:
        raise SystemExit("No training scenarios found under data/training")

    combined = pd.concat(train_data.values(), axis=0, ignore_index=True)
    columns = list(combined.columns)
    encoding = {}
    for column in columns:
        values = combined[column]
        if pd.api.types.is_bool_dtype(values.dtype):
            encoding[column] = {"type": "boolean"}
        elif pd.api.types.is_numeric_dtype(values.dtype):
            mean = values.mean()
            encoding[column] = {"type": "numeric", "fill_value": float(mean) if pd.notna(mean) else 0.0}
        else:
            encoding[column] = {
                "type": "categorical",
                "categories": sorted(values.dropna().astype(str).unique().tolist()),
            }
    # Learn categories from all files before subsampling training rows.
    X = encode_measurements(combined.iloc[::ROW_STRIDE], {"columns": columns, "encoding": encoding})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=0,
    )
    print(f"Training autoencoder on {X_scaled.shape[0]} rows, {X_scaled.shape[1]} columns...")
    model.fit(X_scaled, X_scaled)

    reconstructed = model.predict(X_scaled)
    reconstruction_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold = float(reconstruction_errors.mean() + 3 * reconstruction_errors.std())
    print(f"Training completed. Threshold set to: {threshold:.6f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH
    artifact = {
        "columns": columns,
        "encoding": encoding,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "layers": [{"weights": weights.tolist(), "bias": bias.tolist()}
                   for weights, bias in zip(model.coefs_, model.intercepts_)],
        "threshold": threshold,
    }
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved model to {output_path}")
if __name__ == "__main__":
    train()
