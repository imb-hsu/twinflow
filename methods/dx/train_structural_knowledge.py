"""Train (assemble) the structural-knowledge-based diagnosis model.

Adapts methods/dx/structural_knowledge_based.py to this project's existing
knowledge files instead of a bespoke config format:

- structural_hierarchy.json: area -> component type -> component identifiers.
- component_type_knowledge.json: component type -> faultTypes (possible faults).

The resulting artifact maps a measurement/fault column (e.g. "RC_144.MotorFault")
to its owning component, component type, production area, and the possible
faults for that component type. Applying this model still requires anomaly
flags (e.g. from models/range_monitoring.json) to know *which* variables are
currently anomalous.

Run this script from the repository root:

    python scripts/train_structural_knowledge.py
"""

import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard" / "features"))

import dataset_utils as ds  # noqa: E402

STRUCTURAL_HIERARCHY_PATH = REPO_ROOT / "prior_knowledge" / "structural_hierarchy.json"
COMPONENT_TYPE_KNOWLEDGE_PATH = REPO_ROOT / "prior_knowledge" / "component_type_knowledge.json"


def main() -> None:
    hierarchy = json.loads(STRUCTURAL_HIERARCHY_PATH.read_text(encoding="utf-8"))
    knowledge = json.loads(COMPONENT_TYPE_KNOWLEDGE_PATH.read_text(encoding="utf-8"))

    component_type_by_id: dict[str, str] = {}
    area_by_id: dict[str, str] = {}
    for area_name, component_types in hierarchy.get("areas", {}).items():
        for component_type, component_ids in component_types.items():
            for component_id in component_ids:
                component_type_by_id[component_id] = component_type
                area_by_id[component_id] = area_name

    faults_by_type = {
        component_type: [entry["name"] for entry in entry_data.get("faultTypes", [])]
        for component_type, entry_data in knowledge.items()
    }

    ds.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = ds.MODELS_PATH / "structural_knowledge.joblib"
    joblib.dump(
        {
            "component_type_by_id": component_type_by_id,
            "area_by_id": area_by_id,
            "faults_by_type": faults_by_type,
        },
        output_path,
    )
    print(
        f"Saved structural knowledge for {len(component_type_by_id)} components "
        f"and {len(faults_by_type)} component types to {output_path}"
    )


if __name__ == "__main__":
    main()
