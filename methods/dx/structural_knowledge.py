"""Train (assemble) the structural-knowledge-based diagnosis model.

Adapts methods/dx/structural_knowledge_based.py to this project's existing
knowledge files instead of a bespoke config format:

- structural_hierarchy.json: area -> component type -> component identifiers.
- data/training/*/faults.parquet: fault types observed for each component.

The resulting artifact maps a measurement/fault column (e.g. "RC_144.MotorFault")
to its owning component, component type, production area, and the possible
faults for that component type. Applying this model still requires anomaly
flags (e.g. from models/range_monitoring.json) to know *which* variables are
currently anomalous.

Run this script from the repository root:

    python methods/dx/structural_knowledge.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

METHOD_NAME = "Structural-Knowledge-Based"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODELS_PATH = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = REPO_ROOT / "models" / "structural_knowledge.json"
THRESHOLDS_PATH = REPO_ROOT / "models" / "range_monitoring.json"
from methods.ad.range_monitoring import predict_column_anomalies  # noqa: E402
from scripts.evaluate_methods import extract_fault_labels  # noqa: E402

STRUCTURAL_HIERARCHY_PATH = REPO_ROOT / "prior_knowledge" / "structural_hierarchy.json"


def predict(measurements: pd.DataFrame) -> pd.DataFrame:
    """Diagnose each row using model knowledge and attached range thresholds.

    Load structural knowledge and range thresholds from their predefined JSON files.
    Ambiguous fault types remain unknown; prediction never receives true labels.
    """
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model["thresholds"] = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    violations = predict_column_anomalies(measurements, model["thresholds"])
    results = []
    for row in violations.itertuples(index=False, name=None):
        hit_counts = {}
        for column, hit in zip(violations.columns, row):
            if hit:
                component, _, _ = column.rpartition(".")
                component = component or column
                hit_counts[component] = hit_counts.get(component, 0) + 1
        if not hit_counts:
            results.append(("unknown", "unknown"))
            continue
        component = max(hit_counts, key=hit_counts.get)
        component_type = model["component_type_by_id"].get(component)
        faults = (model["faults_by_component"].get(component, [])
                  if "faults_by_component" in model else model["faults_by_type"].get(component_type, []))
        results.append((component, faults[0] if len(faults) == 1 else "unknown"))
    return pd.DataFrame(results, index=measurements.index, columns=["component", "fault_type"])


def train() -> None:
    hierarchy = json.loads(STRUCTURAL_HIERARCHY_PATH.read_text(encoding="utf-8"))
    training_dir = REPO_ROOT / "data" / "training"
    train_data = {
        p.name: pd.read_parquet(p / "faults.parquet")
        for p in sorted(training_dir.iterdir())
        if p.is_dir()
    }
    if not train_data:
        raise SystemExit("No training scenarios found under data/training")

    component_type_by_id: dict[str, str] = {}
    area_by_id: dict[str, str] = {}
    for area_name, component_types in hierarchy.get("areas", {}).items():
        for component_type, component_ids in component_types.items():
            for component_id in component_ids:
                component_type_by_id[component_id] = component_type
                area_by_id[component_id] = area_name

    faults_by_component = {}
    faults_by_type = {}
    for scenario_name, faults in train_data.items():
        labels = extract_fault_labels(faults, scenario_name)
        for column in labels.columns[labels.any(axis=0)]:
            component, separator, fault_type = column.rpartition(".")
            if not separator:
                raise ValueError(f"{scenario_name}: fault column has no component prefix: {column}")
            faults_by_component.setdefault(component, set()).add(fault_type)
            component_type = component_type_by_id.get(component)
            if component_type is not None:
                faults_by_type.setdefault(component_type, set()).add(fault_type)
    faults_by_component = {component: sorted(faults) for component, faults in faults_by_component.items()}
    faults_by_type = {component_type: sorted(faults) for component_type, faults in faults_by_type.items()}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH
    artifact = {
        "component_type_by_id": component_type_by_id,
        "area_by_id": area_by_id,
        "faults_by_type": faults_by_type,
        "faults_by_component": faults_by_component,
    }
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"Saved structural knowledge for {len(component_type_by_id)} components "
        f"and {len(faults_by_type)} component types to {output_path}"
    )


if __name__ == "__main__":
    train()
