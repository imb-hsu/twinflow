"""
Structural-Knowledge-Based Fault Diagnosis

This module implements a structural knowledge-based approach for fault diagnosis
in cyber-physical production systems. The method uses prior knowledge about the
organization of the production system to localize faults.

The system structure is represented by relationships between:
- Production areas
- Component types
- Individual components
- Observable variables

After an anomaly is detected in one or more signals, the affected variables are
mapped to the corresponding component nodes in the structural knowledge model.
Based on the identified component type, the set of possible fault types can be
retrieved.

This approach does not require an explicit causal model. Instead, it relies on
ownership and hierarchy relations:
    anomalous variable → component → component type → possible faults

Therefore, this method provides an interpretable baseline for mapping detected
anomalies to affected components and candidate fault types.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class Component:
    """Represents an individual component in the production system."""

    id: str
    name: str
    component_type: str
    production_area: str
    observable_variables: List[str] = field(default_factory=list)


@dataclass
class ComponentType:
    """Represents a type of component with associated fault types."""

    name: str
    possible_faults: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class ProductionArea:
    """Represents a production area containing multiple components."""

    id: str
    name: str
    components: List[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Result of structural knowledge-based diagnosis."""

    anomalous_variables: List[str]
    affected_components: List[str]
    component_types: List[str]
    possible_faults: Dict[str, List[str]]  # component_id -> list of fault types
    diagnosis_path: List[Dict[str, str]]  # trace of reasoning steps


class StructuralKnowledgeModel:
    """
    Structural knowledge model for fault diagnosis.

    Maintains the hierarchical structure of the production system and provides
    methods for mapping anomalies to components and fault types.
    """

    def __init__(self):
        self.production_areas: Dict[str, ProductionArea] = {}
        self.components: Dict[str, Component] = {}
        self.component_types: Dict[str, ComponentType] = {}
        self.variable_to_component: Dict[str, str] = {}  # variable -> component_id

    def add_production_area(self, area: ProductionArea) -> None:
        """Add a production area to the model."""
        self.production_areas[area.id] = area

    def add_component_type(self, comp_type: ComponentType) -> None:
        """Add a component type with associated fault types."""
        self.component_types[comp_type.name] = comp_type

    def add_component(self, component: Component) -> None:
        """Add a component and update variable mappings."""
        self.components[component.id] = component

        # Update variable to component mapping
        for var in component.observable_variables:
            self.variable_to_component[var] = component.id

        # Update production area
        if component.production_area in self.production_areas:
            if (
                component.id
                not in self.production_areas[component.production_area].components
            ):
                self.production_areas[component.production_area].components.append(
                    component.id
                )

    def get_component_by_variable(self, variable: str) -> Optional[Component]:
        """Map a variable to its corresponding component."""
        component_id = self.variable_to_component.get(variable)
        if component_id:
            return self.components.get(component_id)
        return None

    def get_possible_faults_for_component(self, component: Component) -> List[str]:
        """Retrieve possible fault types for a component based on its type."""
        comp_type = self.component_types.get(component.component_type)
        if comp_type:
            return comp_type.possible_faults
        return []

    def load_from_config(self, config_path: str) -> None:
        """
        Load structural knowledge from a configuration file.

        Expected JSON structure:
        {
            "production_areas": [...],
            "component_types": [...],
            "components": [...]
        }
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load production areas
        for area_data in config.get("production_areas", []):
            area = ProductionArea(
                id=area_data["id"],
                name=area_data["name"],
                components=area_data.get("components", []),
            )
            self.add_production_area(area)

        # Load component types
        for type_data in config.get("component_types", []):
            comp_type = ComponentType(
                name=type_data["name"],
                possible_faults=type_data.get("possible_faults", []),
                description=type_data.get("description"),
            )
            self.add_component_type(comp_type)

        # Load components
        for comp_data in config.get("components", []):
            component = Component(
                id=comp_data["id"],
                name=comp_data["name"],
                component_type=comp_data["component_type"],
                production_area=comp_data["production_area"],
                observable_variables=comp_data.get("observable_variables", []),
            )
            self.add_component(component)


class StructuralKnowledgeBasedDiagnosis:
    """
    Structural knowledge-based fault diagnosis implementation.

    Uses hierarchical system structure to map detected anomalies to affected
    components and candidate fault types.
    """

    def __init__(self, knowledge_model: StructuralKnowledgeModel):
        self.model = knowledge_model

    def diagnose(self, anomalous_variables: List[str]) -> DiagnosisResult:
        """
        Perform structural knowledge-based diagnosis.

        Args:
            anomalous_variables: List of variable names showing anomalous behavior

        Returns:
            DiagnosisResult containing affected components and possible faults
        """
        affected_components: Set[str] = set()
        component_types: Set[str] = set()
        possible_faults: Dict[str, List[str]] = {}
        diagnosis_path: List[Dict[str, str]] = []

        # Step 1: Map anomalous variables to components
        for variable in anomalous_variables:
            component = self.model.get_component_by_variable(variable)

            if component:
                affected_components.add(component.id)
                component_types.add(component.component_type)

                # Record diagnosis step
                diagnosis_path.append(
                    {
                        "step": "variable_to_component",
                        "variable": variable,
                        "component_id": component.id,
                        "component_name": component.name,
                        "component_type": component.component_type,
                    }
                )

                # Step 2: Retrieve possible faults for the component
                faults = self.model.get_possible_faults_for_component(component)
                if faults:
                    possible_faults[component.id] = faults

                    # Record diagnosis step
                    diagnosis_path.append(
                        {
                            "step": "component_to_faults",
                            "component_id": component.id,
                            "component_type": component.component_type,
                            "possible_faults": ", ".join(faults),
                        }
                    )
            else:
                # Variable not found in model
                diagnosis_path.append(
                    {
                        "step": "variable_not_mapped",
                        "variable": variable,
                        "status": "unknown_variable",
                    }
                )

        return DiagnosisResult(
            anomalous_variables=anomalous_variables,
            affected_components=list(affected_components),
            component_types=list(component_types),
            possible_faults=possible_faults,
            diagnosis_path=diagnosis_path,
        )

    def diagnose_with_ranking(
        self,
        anomalous_variables: List[str],
        anomaly_scores: Optional[Dict[str, float]] = None,
    ) -> DiagnosisResult:
        """
        Perform diagnosis with optional ranking based on anomaly scores.

        Args:
            anomalous_variables: List of variable names showing anomalous behavior
            anomaly_scores: Optional dict mapping variables to anomaly scores

        Returns:
            DiagnosisResult with components ranked by aggregated anomaly scores
        """
        result = self.diagnose(anomalous_variables)

        if anomaly_scores:
            # Aggregate scores by component
            component_scores: Dict[str, float] = {}

            for variable in anomalous_variables:
                component = self.model.get_component_by_variable(variable)
                if component and variable in anomaly_scores:
                    if component.id not in component_scores:
                        component_scores[component.id] = 0.0
                    component_scores[component.id] += anomaly_scores[variable]

            # Sort components by score
            result.affected_components = sorted(
                result.affected_components,
                key=lambda comp_id: component_scores.get(comp_id, 0.0),
                reverse=True,
            )

            # Add ranking info to diagnosis path
            result.diagnosis_path.append(
                {"step": "component_ranking", "scores": str(component_scores)}
            )

        return result


def create_example_model() -> StructuralKnowledgeModel:
    """
    Create an example structural knowledge model for demonstration.

    Returns:
        Populated StructuralKnowledgeModel instance
    """
    model = StructuralKnowledgeModel()

    # Define component types with possible faults
    model.add_component_type(
        ComponentType(
            name="Conveyor",
            possible_faults=[
                "belt_slip",
                "motor_failure",
                "sensor_failure",
                "blockage",
            ],
            description="Belt conveyor for material transport",
        )
    )

    model.add_component_type(
        ComponentType(
            name="Robot6Axis",
            possible_faults=[
                "gripper_failure",
                "positioning_error",
                "communication_loss",
                "collision",
            ],
            description="Six-axis industrial robot",
        )
    )

    model.add_component_type(
        ComponentType(
            name="AMR",
            possible_faults=[
                "navigation_failure",
                "battery_low",
                "communication_loss",
                "obstacle_detection_failure",
            ],
            description="Autonomous Mobile Robot",
        )
    )

    model.add_component_type(
        ComponentType(
            name="Workstation",
            possible_faults=["tool_failure", "quality_issue", "timeout"],
            description="Manual or automated workstation",
        )
    )

    # Define production areas
    model.add_production_area(
        ProductionArea(id="area_1", name="Goods Input", components=[])
    )

    model.add_production_area(
        ProductionArea(id="area_2", name="Robot Stations", components=[])
    )

    model.add_production_area(
        ProductionArea(id="area_3", name="Transport", components=[])
    )

    # Define components with observable variables
    model.add_component(
        Component(
            id="conv_01",
            name="Conveyor_01",
            component_type="Conveyor",
            production_area="area_1",
            observable_variables=[
                "conv_01_speed",
                "conv_01_current",
                "conv_01_vibration",
            ],
        )
    )

    model.add_component(
        Component(
            id="robot_rs1",
            name="Robot_RS1",
            component_type="Robot6Axis",
            production_area="area_2",
            observable_variables=[
                "robot_rs1_pos_x",
                "robot_rs1_pos_y",
                "robot_rs1_pos_z",
                "robot_rs1_gripper_force",
                "robot_rs1_cycle_time",
            ],
        )
    )

    model.add_component(
        Component(
            id="amr_01",
            name="AMR_01",
            component_type="AMR",
            production_area="area_3",
            observable_variables=[
                "amr_01_battery",
                "amr_01_speed",
                "amr_01_position_x",
                "amr_01_position_y",
            ],
        )
    )

    return model


# Example usage
if __name__ == "__main__":
    # Create example model
    model = create_example_model()

    # Initialize diagnosis system
    diagnosis_system = StructuralKnowledgeBasedDiagnosis(model)

    # Example: Diagnose anomalies in conveyor variables
    anomalous_vars = ["conv_01_speed", "conv_01_vibration"]
    result = diagnosis_system.diagnose(anomalous_vars)

    print("Diagnosis Results:")
    print(f"Anomalous variables: {result.anomalous_variables}")
    print(f"Affected components: {result.affected_components}")
    print(f"Component types: {result.component_types}")
    print(f"Possible faults: {result.possible_faults}")
    print("\nDiagnosis path:")
    for step in result.diagnosis_path:
        print(f"  {step}")
