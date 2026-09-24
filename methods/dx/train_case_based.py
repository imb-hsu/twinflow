"""Train the case-based fault diagnosis reference bank.

Adapts methods/dx/case_based.py's approach to the prepared parquet files:
for every boolean fault column, each contiguous "active" interval in a faulty
training scenario contributes one reference case (the measurement vector at
the interval's midpoint, labeled with the fault and its owning component).
A handful of "nominal" cases from 0pct scenarios are added as a no-fault
baseline for the nearest-neighbor search.

Run this script from the repository root:

    python scripts/train_case_based.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard" / "features"))

import dataset_utils as ds  # noqa: E402

NOMINAL_CASES_PER_SCENARIO = 20


def _boolean_fault_columns(faults: pd.DataFrame) -> list[str]:
    return [column for column in faults.columns if str(faults[column].dtype) == "boolean"]


def _fault_segments(active: pd.Series) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) integer-position pairs of contiguous True runs."""
    values = active.fillna(False).to_numpy(dtype=bool)
    segments = []
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _build_case(vector: np.ndarray, fault_label: str) -> dict:
    component_id, _, fault_type = fault_label.rpartition(".")
    return {
        "vector": vector,
        "fault_label": fault_label,
        "component": component_id or fault_label,
        "fault_type": fault_type or fault_label,
    }


def main() -> None:
    datasets = ds.discover_datasets()
    training_scenarios = datasets.get("training", {})

    cases = []
    columns: list[str] | None = None

    for scenario_name in sorted(training_scenarios, key=ds.scenario_sort_key):
        measurements = ds.load_table(datasets, "training", scenario_name, "measurements")
        faults = ds.load_table(datasets, "training", scenario_name, "faults")
        if measurements is None or faults is None:
            continue
        if columns is None:
            columns = ds.numeric_columns(measurements)
        numeric = measurements[columns].fillna(measurements[columns].mean())

        if scenario_name.startswith("0_0pct"):
            # Nominal scenario: sample a few rows as the "no fault" baseline.
            sample_positions = np.linspace(
                0, len(numeric) - 1, num=min(NOMINAL_CASES_PER_SCENARIO, len(numeric)), dtype=int
            )
            for position in sample_positions:
                cases.append(_build_case(numeric.iloc[position].to_numpy(), "nominal"))
            continue

        for fault_column in _boolean_fault_columns(faults):
            for start, end in _fault_segments(faults[fault_column]):
                midpoint = (start + end) // 2
                cases.append(_build_case(numeric.iloc[midpoint].to_numpy(), fault_column))

        print(f"Processed {scenario_name}: {len(cases)} cases so far")

    if not cases or columns is None:
        raise SystemExit("No reference cases could be built from data/training")

    vectors = np.stack([case["vector"] for case in cases])
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    for case, scaled_vector in zip(cases, scaled_vectors):
        case["vector"] = scaled_vector

    ds.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = ds.MODELS_PATH / "case_based.joblib"
    joblib.dump({"cases": cases, "scaler": scaler, "columns": columns}, output_path)
    print(f"Saved {len(cases)} reference cases to {output_path}")


if __name__ == "__main__":
    main()
