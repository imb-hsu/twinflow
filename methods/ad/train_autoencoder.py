"""Train the autoencoder-based anomaly detector.

Adapts methods/ad/autoencoder.py's approach (train on nominal data, flag high
reconstruction error) to the prepared measurements.parquet files. Uses
scikit-learn's MLPRegressor as a bottlenecked encoder/decoder instead of
PyTorch, since torch is not an existing project dependency.

Run this script from the repository root:

    python scripts/train_autoencoder.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard" / "features"))

import dataset_utils as ds  # noqa: E402

NOMINAL_SCENARIO_PREFIX = "0_0pct"
# Bottleneck (8) forces the network to compress before reconstructing its input.
HIDDEN_LAYER_SIZES = (64, 8, 64)
# Subsample rows to keep CPU training time reasonable; consecutive samples are
# highly correlated (0.1s steps), so this barely reduces the information content.
ROW_STRIDE = 5


def main() -> None:
    datasets = ds.discover_datasets()
    training_scenarios = datasets.get("training", {})
    nominal_scenarios = [
        name for name in training_scenarios if name.startswith(NOMINAL_SCENARIO_PREFIX)
    ]
    if not nominal_scenarios:
        raise SystemExit("No 0pct-fault training scenarios found under data/training")

    frames = []
    for scenario_name in nominal_scenarios:
        df = ds.load_table(datasets, "training", scenario_name, "measurements")
        if df is not None:
            frames.append(df)
            print(f"Loaded {scenario_name}: {df.shape[0]} rows")

    combined = pd.concat(frames, axis=0, ignore_index=True).iloc[::ROW_STRIDE]
    columns = ds.numeric_columns(combined)
    X = combined[columns].fillna(combined[columns].mean()).to_numpy()

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

    ds.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = ds.MODELS_PATH / "autoencoder.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "threshold": threshold,
            "columns": columns,
        },
        output_path,
    )
    print(f"Saved model to {output_path}")
if __name__ == "__main__":
    main()
