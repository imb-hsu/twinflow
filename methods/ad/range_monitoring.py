"""
Range Monitoring Method for Anomaly Detection

This module implements a range monitoring approach that estimates min/max thresholds
for each variable in the dataset and uses them for anomaly detection.
"""

import io
import json
import zstandard as zstd
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union, Any
import numpy as np


def _iter_json_objects(file_path: Union[str, Path]) -> Iterator[Any]:
    """
    Stream JSON objects from a compressed file.

    The training files are newline-delimited JSON compressed with zstd. A plain
    json.loads() call fails on that format with "Extra data" because line 2 starts
    a second valid JSON object. This loader keeps memory bounded and also supports
    regular single-document JSON files as a fallback.
    """
    file_path = Path(file_path)
    decoder = json.JSONDecoder()

    with open(file_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            first_line = text_stream.readline()

            if not first_line:
                return

            stripped = first_line.strip()
            if not stripped:
                for line in text_stream:
                    stripped = line.strip()
                    if stripped:
                        first_line = line
                        break
                else:
                    return

            try:
                yield json.loads(first_line)
                for line_number, line in enumerate(text_stream, start=2):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        yield json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON object in {file_path} at line {line_number}: {exc}"
                        ) from exc
            except json.JSONDecodeError:
                # Fallback for a normal JSON document that spans multiple lines.
                remaining_text = first_line + text_stream.read()
                try:
                    obj, end_index = decoder.raw_decode(remaining_text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc

                if remaining_text[end_index:].strip():
                    raise ValueError(
                        f"Invalid JSON in {file_path}: extra data after first JSON value"
                    )

                yield obj


def _iter_records(data: Any) -> Iterator[Dict[str, Any]]:
    """Yield record dictionaries from the data shapes used by the project."""
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("data", "records", "measurements"):
            if key in data:
                records = data[key]
                break
        else:
            records = [data]
    else:
        return

    if isinstance(records, dict):
        records = [records]

    for record in records:
        if isinstance(record, dict):
            yield _flatten_record(record)


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten top-level metadata and the nested `values` payload into one mapping.

    In the data files, `values` is a JSON-encoded string containing a list of
    component dictionaries, e.g. [{"AMP_1": {"AMP_1.CurrentSpeed": "0"}}].
    """
    flattened = {
        key: value
        for key, value in record.items()
        if key != "values" and not isinstance(value, (dict, list))
    }

    values = record.get("values")
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError:
            values = None

    if values is not None:
        flattened.update(_flatten_nested_values(values))

    return flattened


def _flatten_nested_values(value: Any, prefix: str = "") -> Dict[str, Any]:
    flattened = {}

    if isinstance(value, dict):
        for key, nested_value in value.items():
            key = str(key)
            if prefix and key.startswith(f"{prefix}."):
                name = key
            elif prefix:
                name = f"{prefix}.{key}"
            else:
                name = key

            if isinstance(nested_value, (dict, list)):
                flattened.update(_flatten_nested_values(nested_value, name))
            else:
                flattened[name] = nested_value
    elif isinstance(value, list):
        for item in value:
            flattened.update(_flatten_nested_values(item, prefix))

    return flattened


def _to_float(value: Any) -> Optional[float]:
    """Convert numeric scalars and numeric strings from the data files to float."""
    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return None if np.isnan(value) else value

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None

        normalized_lower = normalized.lower()
        if normalized_lower in {"true", "false", "nan", "none", "null"}:
            return None

        # The exported data uses comma decimal separators for some values.
        if "," in normalized and "." not in normalized:
            normalized = normalized.replace(",", ".")

        try:
            value = float(normalized)
        except ValueError:
            return None

        return None if np.isnan(value) else value

    return None


def estimate_thresholds(
    data_files: List[Union[str, Path]],
    percentile_margin: float = 0.0,
    exclude_variables: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Estimate min/max thresholds for each numerical variable in the dataset.

    Args:
        data_files: List of paths to json.zst files containing nominal operation data
        percentile_margin: Optional margin to extend thresholds (0.0 = strict min/max,
                          0.05 = use 5th/95th percentile, etc.)
        exclude_variables: List of variable names to exclude from threshold estimation

    Returns:
        Dictionary mapping variable names to their thresholds:
        {
            "variable_name": {
                "min": float,
                "max": float,
                "mean": float,
                "std": float
            }
        }
    """
    if exclude_variables is None:
        exclude_variables = []

    # Dictionary to accumulate all values for each variable
    variable_values = {}

    # Read all data files
    for file_path in data_files:
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue

        print(f"Processing file: {file_path}")

        # Process each record without loading the full decompressed file in memory.
        for data in _iter_json_objects(file_path):
            # Extract variables from data. Supports JSON-lines files, lists of
            # records, or wrapper dictionaries with common record keys.
            records = _iter_records(data)
            for record in records:
                for var_name, value in record.items():
                    # Skip excluded variables
                    if var_name in exclude_variables:
                        continue

                    value = _to_float(value)
                    if value is None:
                        continue

                    # Initialize list for this variable if needed
                    if var_name not in variable_values:
                        variable_values[var_name] = []

                    variable_values[var_name].append(value)

    # Calculate thresholds for each variable
    thresholds = {}

    for var_name, values in variable_values.items():
        if len(values) == 0:
            continue

        values_array = np.array(values)

        # Calculate statistics
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)

        # Calculate min/max with optional percentile margin
        if percentile_margin > 0:
            lower_percentile = percentile_margin * 100
            upper_percentile = (1 - percentile_margin) * 100
            min_val = np.percentile(values_array, lower_percentile)
            max_val = np.percentile(values_array, upper_percentile)
        else:
            min_val = np.min(values_array)
            max_val = np.max(values_array)

        thresholds[var_name] = {
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(mean_val),
            "std": float(std_val),
        }

    print(f"\nEstimated thresholds for {len(thresholds)} variables")

    return thresholds


def save_thresholds(
    thresholds: Dict[str, Dict[str, float]], output_path: Union[str, Path]
) -> None:
    """
    Save estimated thresholds to a JSON file.

    Args:
        thresholds: Dictionary of variable thresholds
        output_path: Path where to save the thresholds
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"Thresholds saved to: {output_path}")


def load_thresholds(threshold_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load thresholds from a JSON file.

    Args:
        threshold_path: Path to the threshold file

    Returns:
        Dictionary of variable thresholds
    """
    threshold_path = Path(threshold_path)

    with open(threshold_path, "r") as f:
        thresholds = json.load(f)

    print(f"Loaded thresholds for {len(thresholds)} variables from: {threshold_path}")

    return thresholds


def detect_anomalies(
    data_file: Union[str, Path],
    thresholds: Dict[str, Dict[str, float]],
    return_details: bool = False,
) -> Union[List[bool], Dict[str, Any]]:
    """
    Detect anomalies in data using range monitoring.

    Args:
        data_file: Path to json.zst file with data to check
        thresholds: Dictionary of variable thresholds (from estimate_thresholds)
        return_details: If True, return detailed anomaly information

    Returns:
        If return_details is False: List of boolean values indicating anomalies per record
        If return_details is True: Dictionary with anomaly details including:
            - anomaly_flags: List[bool]
            - anomaly_variables: List[List[str]] - which variables were anomalous
            - anomaly_values: List[Dict[str, float]] - what were the anomalous values
    """
    data_file = Path(data_file)

    # Detect anomalies
    anomaly_flags = []
    anomaly_variables = []
    anomaly_values = []

    for data in _iter_json_objects(data_file):
        for record in _iter_records(data):
            is_anomalous = False
            anomalous_vars = []
            anomalous_vals = {}

            for var_name, value in record.items():
                # Skip variables without thresholds
                if var_name not in thresholds:
                    continue

                value = _to_float(value)
                if value is None:
                    continue

                # Check if value is out of range
                min_threshold = thresholds[var_name]["min"]
                max_threshold = thresholds[var_name]["max"]

                if value < min_threshold or value > max_threshold:
                    is_anomalous = True
                    anomalous_vars.append(var_name)
                    anomalous_vals[var_name] = value

            anomaly_flags.append(is_anomalous)
            anomaly_variables.append(anomalous_vars)
            anomaly_values.append(anomalous_vals)

    if return_details:
        return {
            "anomaly_flags": anomaly_flags,
            "anomaly_variables": anomaly_variables,
            "anomaly_values": anomaly_values,
            "total_records": len(anomaly_flags),
            "anomalous_records": sum(anomaly_flags),
        }
    else:
        return anomaly_flags


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).resolve().parents[1]

    # Step 1: Estimate thresholds from nominal operation data
    print("=== Estimating thresholds from nominal data ===")

    nominal_files = list(
        (project_root / "data/training/0pct_faults").glob("*.json.zst")
    )

    if nominal_files:
        # Estimate thresholds (use percentile_margin to make thresholds less strict)
        thresholds = estimate_thresholds(
            nominal_files,
            percentile_margin=0.01,  # Use 1st and 99th percentile
            exclude_variables=["timestamp", "scenario_id"],  # Variables to exclude
        )

        # Save thresholds
        threshold_file = project_root / "thresholds/range_monitoring_thresholds.json"
        save_thresholds(thresholds, threshold_file)

        # Step 2: Use thresholds for anomaly detection
        print("\n=== Detecting anomalies in test data ===")

        # Load thresholds (in practice, you might do this separately)
        loaded_thresholds = load_thresholds(threshold_file)

        # Detect anomalies in test file
        test_file = project_root / "data/test/scenario_faulty_001.json.zst"

        if Path(test_file).exists():
            results = detect_anomalies(
                test_file, loaded_thresholds, return_details=True
            )

            print(f"\nAnomaly Detection Results:")
            print(f"Total records: {results['total_records']}")
            print(f"Anomalous records: {results['anomalous_records']}")
            print(
                f"Anomaly rate: {results['anomalous_records'] / results['total_records'] * 100:.2f}%"
            )

            # Show first few anomalies
            print("\nFirst anomalous records:")
            for i, (is_anom, vars_list, vals_dict) in enumerate(
                zip(
                    results["anomaly_flags"][:10],
                    results["anomaly_variables"][:10],
                    results["anomaly_values"][:10],
                )
            ):
                if is_anom:
                    print(f"  Record {i}: {len(vars_list)} anomalous variables")
                    for var in vars_list[:3]:  # Show first 3 anomalous variables
                        print(f"    - {var}: {vals_dict[var]}")
    else:
        print("No nominal data files specified. Please update the nominal_files list.")
        print("\nExample usage:")
        print("  1. Place your nominal operation json.zst files in a directory")
        print("  2. Update nominal_files list with paths to these files")
        print("  3. Run this script to estimate thresholds")
        print("  4. Use the saved thresholds for anomaly detection on test data")
