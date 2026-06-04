"""
Scenario Generator for HSU TwinFlow Benchmark

This module generates simulation scenario files for the HSU TwinFlow benchmark.
Scenarios include pallet creation events and fault injection actions with
corresponding repair actions. The generator supports configurable fault
percentages, multi-fault scenarios, and area-specific fault injection.

Generated scenarios are saved as JSON files for use in digital twin simulations.
"""

import json
import random
from pathlib import Path
from typing import Any

PALLET_CONFIGURATIONS = [
    "PalletConfiguration-NOVA-6-DTF-6",
    "PalletConfiguration-NOVA-DTF-HSU-Mix",
    "HSU-1Box",
]
NUM_REPETITIONS_TRAINING = 5


def format_time(minutes: float) -> str:
    """
    Convert minutes as float to MM:SS format.

    Args:
        minutes: Time in minutes as a floating-point number.

    Returns:
        Time formatted as "MM:SS" string.
    """
    mm = int(minutes)
    ss = int(round((minutes - mm) * 60))

    if ss == 60:
        mm += 1
        ss = 0

    return f"{mm:02d}:{ss:02d}"


def sample_fault_value(fault_definition: dict, rng: random.Random) -> tuple[Any, Any]:
    """
    Sample fault activation value and return corresponding repair value.

    Args:
        fault_definition: Dictionary containing fault type specification with keys
                         'type', 'min', 'max', and optionally 'repairValue'.
        rng: Random number generator instance for reproducible sampling.

    Returns:
        Tuple of (activation_value, repair_value) where activation_value is the
        sampled fault value and repair_value is the value used to repair the fault.

    Raises:
        ValueError: If fault type is not 'boolean' or 'number'.
    """
    fault_type = fault_definition["type"]

    if fault_type == "boolean":
        return True, fault_definition.get("repairValue", False)

    if fault_type == "number":
        low = fault_definition["min"]
        high = fault_definition["max"]
        activation_value = round(rng.uniform(low, high), 3)
        repair_value = fault_definition.get("repairValue", 0)
        return activation_value, repair_value

    raise ValueError(f"Unsupported fault type: {fault_type}")


def sample_random_fault(
    structural_hierarchy: dict,
    conveyor_segments: dict,
    fault_knowledge: dict,
    rng: random.Random,
    multi: int = 1,
    area: str = None,
) -> list[dict]:
    """
    Sample random faults from the production system.

    Randomly selects area (if not specified), component type, element, fault type,
    and generates activation and repair values. The structural hierarchy defines
    available components organized as: area -> component type -> list of components.
    The fault knowledge defines possible faults per component type.

    Args:
        structural_hierarchy: Dictionary defining system structure with areas and components.
        conveyor_segments: Dictionary mapping conveyor segments to component IDs.
        fault_knowledge: Dictionary defining possible faults per component type.
        rng: Random number generator instance for reproducible sampling.
        multi: Number of faults to sample from the same area (default: 1).
        area: Specific area to sample from, or None for random area selection.

    Returns:
        List of fault dictionaries, each containing area, component_type, element,
        property (fault name), activation_value, and repair_value.
    """

    # Select area randomly if not specified
    if area is None:
        area = rng.choice(list(structural_hierarchy["areas"].keys()))

    faults = []

    # Generate multiple faults from the same area
    for _ in range(multi):
        component_type = rng.choice(list(structural_hierarchy["areas"][area].keys()))

        # Special handling for conveyor components (RC, CC)
        if component_type in ["RC", "CC"]:
            segment = rng.choice(list(conveyor_segments.keys()))
            element_id = rng.choice(conveyor_segments[segment])
        else:
            element_id = rng.choice(structural_hierarchy["areas"][area][component_type])

        fault_definition = rng.choice(fault_knowledge[component_type]["faultTypes"])

        fault_name = fault_definition["name"]

        activation_value, repair_value = sample_fault_value(
            fault_definition,
            rng,
        )

        faults.append(
            {
                "area": area,
                "component_type": component_type,
                "element": element_id,
                "property": fault_name,
                "activation_value": activation_value,
                "repair_value": repair_value,
            }
        )

    return faults


def create_scenario(
    run_id: str,
    pallet_configurations: list[str],
    duration_minutes: int = 60,
    duration_of_fault_intervals_sec: int = 10,
    mean_interarrival_minutes: float = 10.0,
    faulty_time_percentage: float = 0.0,
    multi_fault=1,
    seed: int = 42,
    area: str = None,
) -> dict[str, Any]:
    """
    Create a simulation scenario with pallet creation and fault injection actions.

    Generates a scenario JSON containing pallet creation actions with exponentially
    distributed inter-arrival times and randomly sampled fault intervals. Each fault
    is automatically repaired after the specified fault interval duration.

    Args:
        run_id: Unique identifier for the scenario run.
        pallet_configurations: List of pallet configuration names to randomly select from.
        duration_minutes: Total simulation duration in minutes (default: 60).
        duration_of_fault_intervals_sec: Duration of each fault interval in seconds (default: 10).
        mean_interarrival_minutes: Mean time between pallet arrivals in minutes (default: 10.0).
        faulty_time_percentage: Percentage of simulation time with active faults (0-100) (default: 0.0).
        multi_fault: Number of simultaneous faults per interval (default: 1).
        seed: Random seed for reproducibility (default: 42).
        area: Specific area for fault injection, or None for random selection (default: None).

    Returns:
        Dictionary representing the complete scenario with runId, scenarioId, cycles,
        duration, startDate, and list of actions.

    Raises:
        ValueError: If faulty_time_percentage is not between 0 and 100.
    """

    # Initialize random number generator with seed for reproducibility
    rng = random.Random(seed)
    hierarchy_path = Path("structural_hierarchy.json")
    with hierarchy_path.open("r", encoding="utf-8") as f:
        structural_knowledge = json.load(f)

    conveyor_segments_path = Path("conveyor_segments.json")
    with conveyor_segments_path.open("r", encoding="utf-8") as f:
        conveyor_segments = json.load(f)

    fault_knowledge_path = Path("component_type_knowledge.json")
    with fault_knowledge_path.open("r", encoding="utf-8") as f:
        fault_knowledge = json.load(f)

    actions = []

    # ------------------------------------------------------------
    # Pallet creation actions
    # Generate pallet arrivals using exponential distribution to model
    # realistic inter-arrival times in production systems
    # ------------------------------------------------------------
    t = 0.0

    while True:
        if t > duration_minutes:
            break

        # Randomly select pallet configuration
        configuration_name = rng.choice(pallet_configurations)

        actions.append(
            {
                "element": "PalletCreator",
                "property": "CreatePallet",
                "startsAt": format_time(t),
                "value": configuration_name,
            }
        )

        # Sample next arrival time from exponential distribution
        t += rng.expovariate(1.0 / mean_interarrival_minutes)

    # ------------------------------------------------------------
    # Fault and repair actions
    # Calculate fault intervals based on desired faulty time percentage
    # and ensure non-overlapping fault periods
    # ------------------------------------------------------------
    if not 0 <= faulty_time_percentage <= 100:
        raise ValueError("faulty_time_percentage must be between 0 and 100.")

    # Calculate total faulty time and number of fault intervals
    faulty_minutes = duration_minutes * faulty_time_percentage / 100.0

    number_of_fault_intervals = int(
        faulty_minutes * 60 / duration_of_fault_intervals_sec
    )

    # Generate random fault start times uniformly distributed across simulation duration
    fault_start_minutes = [
        rng.uniform(0, duration_minutes) for _ in range(number_of_fault_intervals)
    ]

    # Sort and filter fault starts to prevent temporal overlap
    fault_start_minutes.sort()
    non_overlapping_starts = []
    previous_fault_end_time = -float("inf")

    for start_time in fault_start_minutes:
        # Only include fault if it starts after previous fault ends
        if start_time >= previous_fault_end_time:
            non_overlapping_starts.append(start_time)
            previous_fault_end_time = (
                start_time + duration_of_fault_intervals_sec / 60.0
            )

    fault_start_minutes = non_overlapping_starts

    for start_minute in fault_start_minutes:
        sampled_faults = sample_random_fault(
            structural_knowledge,
            conveyor_segments,
            fault_knowledge,
            rng,
            multi=multi_fault,
            area=area,
        )

        for sampled_fault in sampled_faults:
            actions.append(
                {
                    "element": sampled_fault["element"],
                    "property": sampled_fault["property"],
                    "startsAt": format_time(start_minute),
                    "value": sampled_fault["activation_value"],
                }
            )

            actions.append(
                {
                    "element": sampled_fault["element"],
                    "property": sampled_fault["property"],
                    "startsAt": format_time(
                        start_minute + duration_of_fault_intervals_sec / 60.0
                    ),
                    "value": sampled_fault["repair_value"],
                }
            )

    actions.sort(key=lambda a: a["startsAt"])

    scenario = {
        "runId": run_id,
        "scenarioId": run_id,
        "cycles": 1,
        "duration": f"{duration_minutes:02d}:00",
        "startDate": "00:00",
        "actions": actions,
    }

    print(f"Created scenario: {run_id}")
    print(
        f"Pallet creation actions: {sum(a['property'] == 'CreatePallet' for a in actions)}"
    )
    print(f"Fault intervals: {number_of_fault_intervals}")
    print(
        f"Faulty time: {number_of_fault_intervals * duration_of_fault_intervals_sec/60} min / {duration_minutes} min"
    )
    return scenario


if __name__ == "__main__":
    # Create directory structure for scenario files
    output_dir = Path("scenarios")
    output_dir.mkdir(exist_ok=True)

    training_dir = output_dir / "training"
    training_dir.mkdir(exist_ok=True)

    test_dir = output_dir / "test"
    test_dir.mkdir(exist_ok=True)

    # Generate training scenarios with multiple repetitions at different fault percentages
    # This provides varied training data for anomaly detection models
    for repetition in range(1, NUM_REPETITIONS_TRAINING + 1):
        percentages = [0.0, 1.0, 10.0]
        for pct in percentages:
            run_id = f"{str(pct).replace('.', '_')}pct_faults_rep_{repetition}"

            scn = create_scenario(
                run_id=run_id,
                pallet_configurations=PALLET_CONFIGURATIONS,
                duration_minutes=60,
                duration_of_fault_intervals_sec=10,
                mean_interarrival_minutes=10.0,
                faulty_time_percentage=pct,
                seed=42 + int(pct * 10) + int(repetition * 100),
            )

            scenario_file = training_dir / f"{run_id}.json"
            scenario_file.write_text(
                json.dumps(scn, indent=2),
                encoding="utf-8",
            )

    # Generate test scenarios with area-specific single faults
    # Each scenario targets a specific production area for evaluation
    pct = 30
    for area in ["WE", "VZ", "AMR", "PAR", "WA"]:
        run_id = f"{area}_{str(pct).replace('.', '_')}pct_faults"

        scn = create_scenario(
            run_id=run_id,
            pallet_configurations=PALLET_CONFIGURATIONS,
            duration_minutes=60,
            duration_of_fault_intervals_sec=10,
            mean_interarrival_minutes=5.0,
            faulty_time_percentage=pct,
            seed=142 + int(pct * 10),
        )

        scenario_file = test_dir / f"{run_id}.json"
        scenario_file.write_text(
            json.dumps(scn, indent=2),
            encoding="utf-8",
        )

        # Generate test scenarios with area-specific multi-fault conditions
        # These scenarios test diagnosis capabilities with simultaneous faults
        pct = 30
        multi_fault = 2
        run_id = f"{area}_multi_{multi_fault}_{str(pct).replace('.', '_')}pct_faults"

        scn = create_scenario(
            run_id=run_id,
            pallet_configurations=PALLET_CONFIGURATIONS,
            duration_minutes=60,
            duration_of_fault_intervals_sec=10,
            mean_interarrival_minutes=5.0,
            faulty_time_percentage=pct,
            multi_fault=multi_fault,
            seed=142 + int(pct * 10) + multi_fault * 1000,
        )

        scenario_file = test_dir / f"{run_id}.json"
        scenario_file.write_text(
            json.dumps(scn, indent=2),
            encoding="utf-8",
        )
