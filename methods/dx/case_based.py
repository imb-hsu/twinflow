"""Train the case-based fault diagnosis reference bank.

Uses all training folders' measurements and fault labels. Numeric, boolean,
and categorical measurements are encoded consistently for training and prediction.
Each active fault interval contributes its last faulty sample as a reference case.
Only fault cases are used for diagnosis; nominal is not a diagnosis class.

Run this script from the repository root:

    python methods/dx/case_based.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

METHOD_NAME = "Case-Based"

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate_methods import extract_fault_labels, diagnosis_events  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "case_based.json"


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


def _build_case(vector: np.ndarray, fault_label: str) -> dict:
    component_id, _, fault_type = fault_label.rpartition(".")
    return {
        "vector": vector,
        "fault_label": fault_label,
        "component": component_id or fault_label,
        "fault_type": fault_type or fault_label,
    }


def predict(measurements: pd.DataFrame) -> pd.DataFrame:
    """Diagnose each measurement row using the saved, scaled reference cases."""
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    # Ignore nominal cases in older saved models as well.
    model["cases"] = [case for case in model["cases"] if all(
        str(case.get(key, "")).lower() != "nominal"
        for key in ("fault_label", "component", "fault_type"))]
    if not model["cases"]:
        raise ValueError("No fault reference cases available; retrain case-based diagnosis.")
    columns = model["columns"]
    missing = set(columns) - set(measurements.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    if measurements.empty:
        return pd.DataFrame(index=measurements.index, columns=["component", "fault_type"])
    numeric = encode_measurements(measurements, model)
    scaled = (numeric - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    neighbors = NearestNeighbors(n_neighbors=1).fit(
        np.stack([case["vector"] for case in model["cases"]])
    )
    indices = neighbors.kneighbors(scaled, return_distance=False)[:, 0]
    return pd.DataFrame(
        [(model["cases"][i]["component"], model["cases"][i]["fault_type"]) for i in indices],
        index=measurements.index, columns=["component", "fault_type"],
    )


def train() -> None:
    training_dir = REPO_ROOT / "data" / "training"
    train_data = {
        p.name: {"measurements": pd.read_parquet(p / "measurements.parquet"),
                 "faults": pd.read_parquet(p / "faults.parquet")}
        for p in sorted(training_dir.iterdir()) if p.is_dir()
    }
    if not train_data:
        raise SystemExit("No training scenarios found under data/training")
    combined = pd.concat([data["measurements"] for data in train_data.values()], ignore_index=True)
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
            encoding[column] = {"type": "categorical", "categories": sorted(values.dropna().astype(str).unique().tolist())}

    cases = []
    for scenario_name, data in train_data.items():
        measurements = data["measurements"]
        faults = data["faults"]
        if measurements.empty or not measurements.index.equals(faults.index):
            raise ValueError(f"{scenario_name}: nonempty measurements and aligned fault indexes are required")
        labels = extract_fault_labels(faults, scenario_name)
        encoded = encode_measurements(measurements.reindex(columns=columns), {"columns": columns, "encoding": encoding})
        for end, fault_column in diagnosis_events(labels):
            cases.append(_build_case(encoded[end], fault_column))
        print(f"Processed {scenario_name}: {len(cases)} cases so far")

    if not cases:
        raise SystemExit("No reference cases could be built from data/training")

    vectors = np.stack([case["vector"] for case in cases])
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    for case, scaled_vector in zip(cases, scaled_vectors):
        case["vector"] = scaled_vector.tolist()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH
    artifact = {"cases": cases, "columns": columns, "encoding": encoding,
                "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}}
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(cases)} reference cases to {output_path}")


if __name__ == "__main__":
    train()
