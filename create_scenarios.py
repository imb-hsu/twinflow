"""
Scenario Generator for HSU TwinFlow Benchmark

This module generates simulation scenario files for the HSU TwinFlow benchmark.
Scenarios include pallet creation events and fault injection actions with
corresponding repair actions. The generator supports configurable fault
percentages, multi-fault scenarios, and area-specific fault injection.

Generated scenarios are saved as JSON files for use in digital twin simulations.
"""

import json
import math
import random
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

PALLET_CONFIGURATIONS = [
    "PalletConfiguration-NOVA-6-DTF-6",
    "PalletConfiguration-NOVA-DTF-HSU-Mix",
    "HSU-1Box",
]
NUM_REPETITIONS_TRAINING = 3
AUTHOR = "HSU TwinFlow Benchmark"


def format_time(minutes: float) -> str:
    """
    Convert minutes as float to MM:SS format.

    Handles rounding of seconds, where values reaching 60 seconds are
    automatically converted to an additional minute.

    Args:
        minutes: Time in minutes as a floating-point number.

    Returns:
        Time formatted as "MM:SS" string with zero-padded values.
    """
    mm = int(minutes)
    ss = int(round((minutes - mm) * 60))

    if ss == 60:
        mm += 1
        ss = 0

    return f"{mm:02d}:{ss:02d}"


def sample_fault_value(fault_definition: dict, rng: random.Random) -> tuple[Any, Any]:
    """
    Sample fault activation value and return the corresponding repair value.

    For boolean faults, returns True as activation and the specified repairValue
    (default: False). For numeric faults, samples a value uniformly from the
    range [min, max] rounded to three decimal places, and returns the specified
    repairValue (default: 0).

    Args:
        fault_definition: Dictionary containing fault type specification with keys
                         'type' (either 'boolean' or 'number'), 'min', 'max',
                         and optionally 'repairValue'.
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
    and generates activation and repair values. When multi > 1, samples multiple
    independent faults from the same area with potentially different component types
    and fault types. The structural hierarchy defines available components organized
    as: area -> component type -> list of components. The fault knowledge defines
    possible faults per component type.

    Args:
        structural_hierarchy: Dictionary defining system structure with areas and components.
        conveyor_segments: Dictionary mapping conveyor segments to component IDs for
                          conveyor components (RC, CC).
        fault_knowledge: Dictionary defining possible faults per component type.
        rng: Random number generator instance for reproducible sampling.
        multi: Number of independent faults to sample from the same area (default: 1).
        area: Specific area to sample from, or None for random area selection.

    Returns:
        List of fault dictionaries, each containing 'area', 'component_type', 'element'
        (component ID), 'property' (fault name), 'activation_value', and 'repair_value'.
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
    scenario_id: str,
    pallet_configurations: list[str],
    duration_minutes: int = 60,
    duration_of_fault_intervals_sec: float = 30,
    num_pallets: int = 10,
    faulty_time_percentage: float = 0.0,
    multi_fault=1,
    seed: int = 42,
    area: str = None,
    ramping_min: float = 5.0,
    min_normal_between_faults: float = 30.0,
) -> dict[str, Any]:
    """
    Create a simulation scenario with pallet creation and fault injection actions.

    Generates a scenario JSON containing pallet creation actions at one-second intervals
    and randomly sampled fault intervals starting after a ramping period. Each fault
    is automatically repaired after the specified fault interval duration. Fault start
    times are sampled using an exponential distribution where the mean inter-arrival time
    is calculated from the desired faulty_time_percentage and the total number of fault
    intervals needed to achieve that percentage. The actual faulty time percentage may
    differ slightly from the target due to rounding of fault interval counts and the
    stochastic nature of exponential sampling.

    Args:
        scenario_id: Unique identifier for the scenario run.
        pallet_configurations: List of pallet configuration names to randomly select from.
        duration_minutes: Total simulation duration in minutes (default: 60).
        duration_of_fault_intervals_sec: Duration of each fault interval in seconds (default: 30).
        num_pallets: Number of pallets to generate at one-second intervals (default: 10).
        faulty_time_percentage: Target percentage of simulation time with active faults,
                               range 0-100 (default: 0.0). Actual percentage may vary.
        multi_fault: Number of simultaneous faults per interval (default: 1).
        seed: Random seed for reproducibility (default: 42).
        area: Specific area for fault injection, or None for random selection (default: None).
        ramping_min: Minutes to wait before starting fault injection (default: 5.0).
        min_normal_between_faults: Minimum normal operation time between faults in minutes
                                  (currently not enforced, default: 30.0).

    Returns:
        Dictionary containing 'scenario' metadata (id, duration, cycles, etc.) and 'actions'
        list with all pallet creation, fault activation, and repair actions sorted by time.

    Raises:
        ValueError: If faulty_time_percentage is not between 0 and 100.
    """

    print(f"\n*** Creating scenario: {scenario_id}")

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
    # Generate pallet arrivals at one-second intervals starting from 00:00
    # ------------------------------------------------------------
    for i in range(num_pallets):
        t = i / 60.0  # Convert seconds to minutes

        # Randomly select pallet configuration
        configuration_name = rng.choice(pallet_configurations)

        actions.append(
            {
                "area": "WE",
                "element": "PalletCreator",
                "property": "CreatePallet",
                "startsAt": format_time(t),
                "value": configuration_name,
            }
        )

    # ------------------------------------------------------------
    # Fault and repair actions
    # Calculate fault intervals based on desired faulty time percentage
    # and ensure non-overlapping fault periods starting after ramping period
    # ------------------------------------------------------------
    if not 0 <= faulty_time_percentage <= 100:
        raise ValueError("faulty_time_percentage must be between 0 and 100.")

    # Calculate total faulty time and number of fault intervals
    available_fault_duration = duration_minutes - ramping_min
    faulty_minutes = duration_minutes * faulty_time_percentage / 100.0

    number_of_fault_intervals = int(
        math.ceil(faulty_minutes * 60 / duration_of_fault_intervals_sec)
    )

    # Calculate mean time between fault starts using exponential distribution
    # Mean = (total_available_time - total_fault_duration) / number_of_intervals
    if faulty_time_percentage > 0:
        fault_duration_min = duration_of_fault_intervals_sec / 60.0
        total_fault_duration = number_of_fault_intervals * fault_duration_min
        total_normal_time = available_fault_duration - total_fault_duration

        if number_of_fault_intervals > 0 and total_normal_time > 0:
            mean_between_faults = total_normal_time / number_of_fault_intervals
        else:
            mean_between_faults = available_fault_duration
    else:
        fault_duration_min = 0
        mean_between_faults = 0

    # Generate fault start times using exponential distribution
    fault_start_minutes = []
    current_time = ramping_min

    if mean_between_faults > 0:
        while True:
            # Sample inter-arrival time from exponential distribution
            inter_arrival = (
                rng.expovariate(1.0 / mean_between_faults)
                if mean_between_faults > 0
                else 0
            )
            # print(inter_arrival)
            current_time += inter_arrival

            # Stop if we exceed simulation duration
            if current_time >= duration_minutes - fault_duration_min:
                break

            fault_start_minutes.append(current_time)
            # Move past the fault duration for next fault
            current_time += fault_duration_min

    # Sort and filter fault starts to prevent temporal overlap and respect minimum normal time
    fault_start_minutes.sort()

    for start_minute in fault_start_minutes:
        sampled_faults = sample_random_fault(
            structural_knowledge,
            conveyor_segments,
            fault_knowledge,
            rng,
            multi=multi_fault,
            area=area,
        )

        def to_string(value):
            if isinstance(value, bool):
                return str(value).lower()
            return str(value).replace(".", ",")

        for sampled_fault in sampled_faults:
            actions.append(
                {
                    "area": sampled_fault["area"],
                    "element": sampled_fault["element"],
                    "property": sampled_fault["property"],
                    "startsAt": format_time(start_minute),
                    "value": to_string(sampled_fault["activation_value"]),
                }
            )

            actions.append(
                {
                    "area": sampled_fault["area"],
                    "element": sampled_fault["element"],
                    "property": sampled_fault["property"],
                    "startsAt": format_time(
                        start_minute + duration_of_fault_intervals_sec / 60.0
                    ),
                    "value": to_string(sampled_fault["repair_value"]),
                }
            )

    actions.sort(key=lambda a: a["startsAt"])

    now = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    scenario = {
        "author": AUTHOR,
        "createdAt": now,
        "cycles": 1,
        "duration": f"{duration_minutes:02d}:00",
        "id": scenario_id,
        "isAdmin": False,
        "startDate": "00:00",
        "updatedAt": now,
        "version": "",
    }

    actual_number_of_fault_intervals = len(fault_start_minutes)

    print(f"*** Created scenario: {scenario_id}")
    print(
        f"Pallet creation actions: {sum(a['property'] == 'CreatePallet' for a in actions)}"
    )
    print(f"Fault intervals: {actual_number_of_fault_intervals}")
    print(
        f"Faulty time: {actual_number_of_fault_intervals * duration_of_fault_intervals_sec / 60} min / {duration_minutes} min"
    )

    if faulty_time_percentage > 0:
        actual_faulty_time_pct = (
            actual_number_of_fault_intervals
            * duration_of_fault_intervals_sec
            / 60
            / duration_minutes
        ) * 100
        print(f"Actual faulty time percentage: {actual_faulty_time_pct:.2f}%")
        print(f"Mean inter-arrival time between faults: {mean_between_faults:.2f} min")

    return {"scenario": scenario, "actions": actions}


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
            scenario_id = f"{str(pct).replace('.', '_')}pct_faults_rep_{repetition}"

            scn = create_scenario(
                scenario_id=scenario_id,
                pallet_configurations=PALLET_CONFIGURATIONS,
                duration_minutes=60,
                duration_of_fault_intervals_sec=30,
                num_pallets=20,
                ramping_min=5,
                faulty_time_percentage=pct,
                seed=42 + int(pct * 10) + int(repetition * 100),
            )

            scenario_file = training_dir / f"{scenario_id}.json"
            scenario_file.write_text(
                json.dumps(scn, indent=2),
                encoding="utf-8",
            )

    # Generate test scenarios with area-specific single faults
    # Each scenario targets a specific production area for evaluation
    pct = 30
    for area in ["WE", "VZ", "AMR", "PAR", "WA"]:
        scenario_id = f"{area}_{str(pct).replace('.', '_')}pct_faults"

        scn = create_scenario(
            scenario_id=scenario_id,
            pallet_configurations=PALLET_CONFIGURATIONS,
            duration_minutes=60,
            duration_of_fault_intervals_sec=30,
            num_pallets=10,
            faulty_time_percentage=pct,
            area=area,
            ramping_min=5,
            min_normal_between_faults=30,
            seed=142 + int(pct * 10),
        )

        scenario_file = test_dir / f"{scenario_id}.json"
        scenario_file.write_text(
            json.dumps(scn, indent=2),
            encoding="utf-8",
        )

        # Generate test scenarios with area-specific multi-fault conditions
        # These scenarios test diagnosis capabilities with simultaneous faults
        pct = 30
        multi_fault = 2
        scenario_id = (
            f"{area}_multi_{multi_fault}_{str(pct).replace('.', '_')}pct_faults"
        )

        scn = create_scenario(
            scenario_id=scenario_id,
            pallet_configurations=PALLET_CONFIGURATIONS,
            duration_minutes=60,
            duration_of_fault_intervals_sec=30,
            num_pallets=10,
            faulty_time_percentage=pct,
            multi_fault=multi_fault,
            area=area,
            ramping_min=5,
            min_normal_between_faults=30,
            seed=142 + int(pct * 10) + multi_fault * 1000,
        )

        scenario_file = test_dir / f"{scenario_id}.json"
        scenario_file.write_text(
            json.dumps(scn, indent=2),
            encoding="utf-8",
        )
