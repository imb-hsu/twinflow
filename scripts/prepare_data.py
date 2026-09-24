"""Prepare raw compressed simulation exports for training.

Run this script from the repository root:

    python scripts/prepare_data.py

Data paths are resolved relative to this file, regardless of the working directory.

It reads all JSONL samples from each ``.json.zst`` file in
``data/training`` and ``data/test``. For every input, it creates a
same-named directory containing ``measurements.parquet``, ``faults.parquet``,
and ``parameters.parquet``. Each parquet file uses ``simulationTime`` as its
index.

Anonymous Lift entries are assigned using ``LIFT_ORDER`` parameter, producing
``CLT_*.Lift.*`` columns.
AMP entries are assigned to the AMR with the same numeric suffix, producing
``AMR_*.AMP.*`` columns.

Vectors expand into numeric .X/.Y/.Z columns. Boolean values remain booleans,
categorical labels remain strings, and blank values become missing values.

EXCLUDED_COLUMNS and EXCLUDED_COLUMN_PREFIXES are used to filter out unwanted columns from the input data.
"""

import io
from functools import lru_cache
import re
import sys
import traceback
from pathlib import Path
import json
import pandas as pd
import zstandard as zstd


# Recognize exported numbers with decimal commas and optional thousands separators.
GERMAN_NUMBER_PATTERN = re.compile(
    r"""
    ^[+-]?                              # optional sign
    (?:
        \d{1,3}(?:\.\d{3})+            # 1.234 or 1.234.567
        |
        \d+                             # 0, 123, 1234
    )
    (?:,\d+)?                           # optional decimal part
    (?:[eE][+-]?\d+)?                   # optional exponent
    $
    """,
    re.VERBOSE,
)


# Drop export metadata and component groups that are outside this training dataset.
EXCLUDED_COLUMNS = {"runId", "scenarioId"}
EXCLUDED_COLUMN_PREFIXES = ("DTF", "NOVA", "HSU", "Pallet")
# CLT owner of each anonymous Lift, in Lift export order. Derived by cross-correlating
# each Lift's CurrentState transitions against every CLT's CurrentState transitions
# (small time-lag window) across multiple recordings and taking the best 1:1 assignment.
LIFT_ORDER = [
    "CLT_125",
    "CLT_89",
    "CLT_96",
    "CLT_47",
    "CLT_59",
    "CLT_40",
    "CLT_137",
    "CLT_108",
    "CLT_28",
    "CLT_15",
]
# Anchor input paths to the repository so IDE and terminal launches behave alike.
REPO_ROOT = Path(__file__).resolve().parent.parent
DIRS_TO_TRANSFORM = [REPO_ROOT / "data" / "training", REPO_ROOT / "data" / "test"]


# The shared component catalog supplies fault names for classifying columns.
COMPONENT_TYPE_KNOWLEDGE_PATH = REPO_ROOT / "prior_knowledge" / "component_type_knowledge.json"


def read_json_zst(input_file: str | Path) -> pd.DataFrame:
    """Read all samples from a compressed .zst JSONL export into a DataFrame."""
    if isinstance(input_file, str):
        input_file = Path(input_file)
    # Decompression streams samples, but the resulting rows are collected in memory.
    rows = []
    all_columns = set()

    # Decompress bytes incrementally and decode one JSON object per text line.
    with input_file.open("rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as reader:
                for line_number, line in enumerate(reader, start=1):
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    # Arrays and scalar JSON values are not simulation sample records.
                    if not isinstance(sample, dict):
                        raise ValueError(
                            f"Expected a JSON object on line {line_number} in {input_file}"
                        )
                    row = flatten_sample(sample)
                    rows.append(row)
                    # Keep columns that appear only in later samples.
                    # Pandas fills missing cells when constructing the dataframe.
                    all_columns.update(row.keys())

    # Empty exports produce an empty dataframe.
    if not rows:
        return pd.DataFrame()

    # Sort signal names for stable output and use simulation time as the row index.
    metadata_columns = ["simulationTime"]
    variable_columns = sorted(c for c in all_columns if c not in metadata_columns)

    return pd.DataFrame(rows, columns=metadata_columns + variable_columns).set_index(
        "simulationTime"
    )


@lru_cache(maxsize=16384)
def parse_export_value(value: str):
    """Convert a scalar string, reusing results for recurring export values."""
    # The bounded cache avoids repeating numeric regex checks for constants and
    # common sensor values without retaining every distinct value in long runs.
    stripped = value.strip()
    if stripped == "True":
        return True
    if stripped == "False":
        return False
    if stripped in {"", "?", "-?", "+?"} or stripped.casefold() == "nan":
        return float("nan")
    if GERMAN_NUMBER_PATTERN.fullmatch(stripped):
        # For example, "1.234,5" becomes the float 1234.5.
        return float(stripped.replace(".", "").replace(",", "."))
    return value


def flatten_sample(sample: dict) -> dict:
    """Flatten one simulation sample into a dictionary of column values."""
    # Preserve the timestamp separately from the nested component measurements.
    row = {
        "simulationTime": sample.get("simulationTime"),
    }

    values = sample.get("values", [])

    # In the exported files, values is usually a JSON-encoded string.
    if isinstance(values, str):
        values = json.loads(values)

    # Use the configured order, independently of the CLT entries' order.
    lift_count = sum(name == "Lift" for entry in values for name in entry)
    # Reject ambiguous assignments instead of attaching Lift signals to wrong owners.
    if lift_count and (
        lift_count != len(LIFT_ORDER)
        or len(set(LIFT_ORDER)) != len(LIFT_ORDER)
        or any(not re.fullmatch(r"CLT_\d+", name) for name in LIFT_ORDER)
    ):
        raise ValueError(
            f"Cannot assign {lift_count} anonymous Lifts to "
            f"LIFT_ORDER={LIFT_ORDER!r}: expected one Lift per unique CLT"
        )
    lift_owners = iter(LIFT_ORDER)

    for entity_entry in values:
        for entity_name, variables in entity_entry.items():
            # All variables of one anonymous Lift share the same configured owner.
            lift_owner = next(lift_owners) if entity_name == "Lift" else None
            for k, v in variables.items():
                if lift_owner is not None:
                    if not k.startswith("Lift."):
                        raise ValueError(f"Unexpected anonymous Lift variable: {k!r}")
                    # Qualify Lift names so different conveyors get distinct columns.
                    k = f"{lift_owner}.{k}"
                # AMP_n is the platform belonging to AMR_n. Keep its signals
                # distinct from the robot's own speed, state, and fault signals.
                # Most columns are not AMP signals; skip their regex check entirely.
                if k.startswith("AMP_"):
                    amp_match = re.fullmatch(r"AMP_(\d+)\.(.+)", k)
                    if amp_match:
                        k = f"AMR_{amp_match[1]}.AMP.{amp_match[2]}"
                if k in EXCLUDED_COLUMNS or k.startswith(EXCLUDED_COLUMN_PREFIXES):
                    continue

                # Do not silently overwrite a Lift signal already present in this sample.
                if ".Lift." in k and k in row:
                    raise ValueError(f"Duplicate Lift variable: {k!r}")
                if ".AMP." in k and k in row:
                    raise ValueError(f"Duplicate AMP variable: {k!r}")

                # Reuse conversions for values repeated across signals and samples.
                if isinstance(v, str):
                    v = parse_export_value(v)
                row[k] = v
    return row


@lru_cache(maxsize=16384)
def parse_export_vector(value: str) -> tuple[float, float, float]:
    """Decode three German-formatted coordinates separated by a dot and space."""
    text = value.strip()
    if not (text.startswith("<") and text.endswith(">")):
        raise ValueError(f"Invalid vector: {value!r}")
    parts = re.split(r"\.\s+", text[1:-1])
    if len(parts) != 3:
        raise ValueError(f"Expected three vector coordinates: {value!r}")
    result = tuple(parse_export_value(part) for part in parts)
    if any(not isinstance(number, (int, float)) for number in result):
        raise ValueError(f"Invalid vector coordinates: {value!r}")
    return result


def normalize_dataframe(df: pd.DataFrame, knowledge: dict) -> pd.DataFrame:
    """Expand vectors and preserve numeric, boolean, and categorical value types."""
    converted = {}
    for column in df.columns:
        series = df[column]
        component, _, variable = str(column).partition(".")
        component_type = component.split("_", 1)[0]
        entries = (
            entry
            for group in knowledge.get(component_type, {}).values()
            for entry in group
        )
        entry = next((entry for entry in entries if entry["name"] == variable), {})
        if pd.api.types.is_bool_dtype(series.dtype):
            converted[column] = series.astype("boolean")
            continue
        if series.isna().all() and entry.get("type") in {"boolean", "categorical"}:
            dtype = "boolean" if entry["type"] == "boolean" else "string"
            converted[column] = series.astype(dtype)
            continue
        if pd.api.types.is_numeric_dtype(series.dtype):
            converted[column] = series
            continue

        # Also normalize object columns containing booleans, numbers, or blanks.
        series = series.map(lambda value: parse_export_value(value) if isinstance(value, str) else value)
        present = series.dropna()
        if any(isinstance(value, str) and value.strip().startswith("<") for value in present):
            # Expand vectors without losing their direction or magnitude.
            names = [f"{column}.{axis}" for axis in "XYZ"]
            if any(name in df.columns for name in names):
                raise ValueError(f"Vector output columns already exist for {column}")
            try:
                vectors = series.map(
                    lambda value: (float("nan"),) * 3 if pd.isna(value) else parse_export_vector(value)
                )
                expanded = pd.DataFrame(vectors.tolist(), index=df.index, columns=names, dtype=float)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Cannot convert vector column {column}: {error}") from error
            converted.update({name: expanded[name] for name in names})
            print(f"Expanded vector {column} into {', '.join(names)}")
            continue

        # Preserve boolean columns even when missing values give them object dtype.
        if pd.api.types.infer_dtype(present, skipna=True) == "boolean" or (
            present.empty and entry.get("type") == "boolean"
        ):
            converted[column] = series.astype("boolean")
            continue

        # Nullable strings retain labels and missing values without category coding.
        if any(isinstance(value, str) for value in present) or (
            present.empty and entry.get("type") == "categorical"
        ):
            converted[column] = series.astype("string")
            continue

        # Raise on unsupported values; never silently discard data through coercion.
        try:
            converted[column] = pd.to_numeric(series, errors="raise").astype("Float64")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Cannot convert {column} to numeric values: {error}") from error

    return pd.DataFrame(converted, index=df.index)


def split_dataframe(df: pd.DataFrame, output_dir: Path|str) -> None:
    """Split a dataframe into measurement, fault, and parameter parquet files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Load the catalog used to identify faults and configurable parameters.
    with COMPONENT_TYPE_KNOWLEDGE_PATH.open(encoding="utf-8") as file:
        component_type_knowledge = json.load(file)

    # Normalize before classification so every output group preserves value types.
    df = normalize_dataframe(df, component_type_knowledge)

    fault_names = {
        fault_type["name"]
        for component_knowledge in component_type_knowledge.values()
        for fault_type in component_knowledge["faultTypes"]
    }
    fault_endings = tuple(f".{fault_name}".casefold() for fault_name in sorted(fault_names))

    # Parameter names are scoped to their component type. Preserve nested names
    # such as AMP.Height and Lift.Height when matching the catalog.
    parameters_by_type = {
        component_type.casefold(): {
            parameter["name"].casefold()
            for parameter in knowledge["configurableParameters"]
        }
        for component_type, knowledge in component_type_knowledge.items()
    }

    # Identify fault columns, case-insensitively
    fault_cols = [
        col for col in df.columns
        if str(col).casefold().endswith(fault_endings)
    ]

    # Split AMR_1.AMP.Height into type AMR and parameter AMP.Height.
    # Unknown types and names remain measurements unless classified as faults.
    parameter_cols = []
    for col in df.columns:
        component_id, separator, parameter_name = str(col).partition(".")
        component_type = component_id.split("_", 1)[0].casefold()
        if separator and parameter_name.casefold() in parameters_by_type.get(
            component_type, set()
        ):
            parameter_cols.append(col)

    # Copy each output group; measurements contain neither faults nor parameters.
    # Fault and parameter selections are independent and can overlap.
    faults = df[fault_cols].copy()
    parameters = df[parameter_cols].copy()
    data = df.drop(columns=fault_cols + parameter_cols).copy()

    print(f"Regular dataframe shape: {data.shape}")
    print(f"Faults dataframe shape: {faults.shape}")
    print(f"Parameters dataframe shape: {parameters.shape}")

    print("Categorical columns saved as strings:")
    for column in df.select_dtypes(include=["string", "object", "category"]):
        print(f"  {column}: unique values = {df[column].unique()}")

    # Diagnostics: parameters are expected to stay constant within an export.
    print("Parameters with a non-null distinct value count other than 1:")
    for c in parameter_cols:
        if parameters[c].nunique() != 1:
            print(f"  {c}: distinct non-null values = {parameters[c].nunique()}")

    print("Constant measurement columns (one distinct non-null value):")
    # Report constant measurements that may carry little training information.
    for c in data:
        if data[c].nunique() == 1:
            print(f"  {c}: distinct non-null values = {data[c].nunique()}")

    # Retain simulationTime in every output so the three groups can be aligned.
    data.to_parquet(output_dir / "measurements.parquet", index=True)
    faults.to_parquet(output_dir / "faults.parquet", index=True)
    parameters.to_parquet(output_dir / "parameters.parquet", index=True)


def main() -> None:
    # Process training and test exports separately, in deterministic filename order.
    for dir_tr in DIRS_TO_TRANSFORM:
        input_files = sorted(
            path
            for path in Path(dir_tr).iterdir()
            if path.is_file() and path.name.endswith(".json.zst")
        )

        processed_files = 0
        failed_files = 0

        for input_file in input_files:
            try:
                # Store outputs beside the archive in a directory without .json.zst.
                output_dir = input_file.with_name(input_file.name.removesuffix(".json.zst"))
                df = read_json_zst(input_file)
                # Reapply exclusions before splitting the final output columns.
                df = df.loc[
                    :,
                    [
                        column
                        for column in df.columns
                        if column not in EXCLUDED_COLUMNS
                        and not column.startswith(EXCLUDED_COLUMN_PREFIXES)
                    ],
                ]
                output_dir.mkdir(exist_ok=True)
                split_dataframe(df, output_dir)
                print(f"Saved prepared parquet files to {output_dir}")
                processed_files += 1
            except Exception as error:
                # Report a failed export with its traceback and continue with the next.
                failed_files += 1
                print(
                    f"Failed to process {input_file}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                traceback.print_exc()

        # Summarize this directory, including files that failed conversion.
        print(
            f"Finished processing {processed_files} of {len(input_files)} files "
            f"({failed_files} failed)."
        )


# Importing helpers must not trigger conversion of the full dataset.
if __name__ == "__main__":
    main()
