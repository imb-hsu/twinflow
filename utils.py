import io
import itertools
import json
import re
from pathlib import Path
import pandas as pd
import zstandard as zstd


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

EXCLUDED_COLUMNS = {"runId", "scenarioId"}
EXCLUDED_COLUMN_PREFIXES = ("DTF", "NOVA", "HSU", "Pallet")


def should_skip_column(column: str) -> bool:
    """Return whether a column should be omitted from the result."""
    return column in EXCLUDED_COLUMNS or column.startswith(EXCLUDED_COLUMN_PREFIXES)


def read_zst_text(path: Path, max_samples: int | None = -1) -> str:
    """Read up to ``max_samples`` complete JSONL samples from a .zst file."""
    with path.open("rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as reader:
                lines = reader if max_samples is None else itertools.islice(
                    reader, max_samples
                )
                return "".join(lines)


def iter_json_samples_from_text(text: str):
    """
    Yield JSON samples from one of these formats:
    - JSONL: one JSON object per line
    - JSON array: [{...}, {...}]
    - single JSON object: {...}
    """
    text = text.strip()

    if not text:
        return

    if text.startswith("["):
        samples = json.loads(text)
        yield from samples
        return

    if text.startswith("{"):
        lines = text.splitlines()

        if len(lines) > 1:
            for line in lines:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            yield json.loads(text)
        return

    raise ValueError("Unsupported JSON format.")


def flatten_sample(sample: dict) -> dict:
    """Flatten one simulation sample into one CSV row."""
    row = {
        "simulationTime": sample.get("simulationTime"),
    }

    values = sample.get("values", [])

    # In the exported files, values is usually a JSON-encoded string.
    if isinstance(values, str):
        values = json.loads(values)

    for entity_entry in values:
        for _, variables in entity_entry.items():
            for k, v in variables.items():
                if should_skip_column(k):
                    continue

                if isinstance(v, str):
                    stripped = v.strip()

                    if stripped == "True":
                        v = True
                    elif stripped == "False":
                        v = False
                    elif stripped in {"?", "-?", "+?"} or stripped.casefold() == "nan":
                        v = float("nan")
                    elif GERMAN_NUMBER_PATTERN.fullmatch(stripped):
                        normalized = stripped.replace(".", "").replace(",", ".")
                        v = float(normalized)
                row[k] = v
    return row


def process_json_text(text: str) -> pd.DataFrame:
    """Process JSON text and return a DataFrame."""
    rows = []
    all_columns = set()

    for sample in iter_json_samples_from_text(text):
        row = flatten_sample(sample)
        rows.append(row)
        all_columns.update(row.keys())

    if not rows:
        return pd.DataFrame()

    metadata_columns = ["simulationTime"]

    variable_columns = sorted(c for c in all_columns if c not in metadata_columns)
    columns = metadata_columns + variable_columns

    return pd.DataFrame(rows, columns=columns).set_index("simulationTime")



def expand_vector_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    result = df.copy()

    for column in columns:
        components = (
            result[column]
            .astype("string")
            .str.strip("<>")
            .str.split(r"\.\s*", n=2, expand=True)
        )

        if components.shape[1] != 3:
            raise ValueError(f"{column!r} does not consistently contain 3 values")

        components = components.apply(
            lambda values: pd.to_numeric(
                values.str.replace(",", ".", regex=False),
                errors="coerce",
            )
        ).astype("float64")

        components.columns = [
            f"{column}.X",
            f"{column}.Y",
            f"{column}.Z",
        ]

        result[components.columns] = components
    result = result.drop(columns=columns)
    return result


def split_dataframe(df: pd.DataFrame, output_dir: Path) -> None:
    """Split a dataframe into measurement, fault, and parameter parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keywords commonly used for component fault variables
    fault_keywords = [
        "fault",
        "failure",
        "failed",
        "error",
        "alarm",
        "malfunction",
        "defect",
        "breakdown",
        "wrongcalibration",
        "emergencystop",
        "chargeempty"
    ]

    parameter_keywords = [
        r"(?!AMR).*\.deceleration",
        r".*\.acceleration",
        r".*\.resistance",
        r".*rollerdamping",
        r".*targetspeed",
        r".*maximumtorque",
        r"cc.*\.height",
        r"rc.*\.height",
        r"cc.*\.width",
        r"rc.*\.width",
        r".*limit",
        r".*length",
        r".*width",
        r"(?!(?:AMP|Lift)).*height",
        r".*processtime",
        r".*pe1",
        r".*pe2",
        r".*pe3",
        r".*pe4",
        r".*mass",
        r"(CC|RC|CLT).*currentdirection",
        r".*maximumforce",
        r".*worldy",
    ]

    # Identify fault columns, case-insensitively
    fault_cols = [
        col for col in df.columns
        if any(keyword in str(col).lower() for keyword in fault_keywords)
    ]

    # Identify parameter columns, case-insensitively
    parameter_cols = [
        col for col in df.columns
        if any(re.compile(keyword, re.IGNORECASE).fullmatch(str(col)) for keyword in parameter_keywords)
    ]

    # Split the original dataframe by columns
    faults = df[fault_cols].copy()
    parameters = df[parameter_cols].copy()
    data = df.drop(columns=fault_cols + parameter_cols).copy()

    print(f"Regular dataframe shape: {data.shape}")
    print(f"Faults dataframe shape: {faults.shape}")
    print(f"Parameters dataframe shape: {parameters.shape}")

    mapping = {
        "Forwards": 0.0,
        "Reverse": 1.0,
    }

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        values = set(df[column].dropna().unique())

        if values == set(mapping):
            df[column] = df[column].map(mapping).astype("Float64")

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    category_counts = df[categorical_columns].nunique()

    for i, cc in enumerate(category_counts):
        if cc > 2:
            print(categorical_columns[i])
            print(df[categorical_columns[i]].unique())

    print("Parameter")
    for c in parameter_cols:
        if parameters[c].nunique() != 1:
            print(c)
            print(parameters[c].nunique())

    print("Data")
    for c in data:
        if data[c].nunique() == 1:
            print(c)
            print(data[c].nunique())
    print("Data")

    for c in data:
        if data[c].dtype not in [float, bool]:
            print(c)
            print(data[c].dtype)
            data[c] = data[c].astype(str)
            print(data[c].head())

    data.to_parquet(output_dir / "measurements.parquet", index=True)
    faults.to_parquet(output_dir / "faults.parquet", index=True)
    parameters.to_parquet(output_dir / "parameters.parquet", index=True)