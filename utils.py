import csv
import json
import warnings
from pathlib import Path

import pandas as pd
import zstandard as zstd


def read_zst_text(path: Path) -> str:
    """Read and decompress a .zst text file."""
    with path.open("rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            return reader.read().decode("utf-8")


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
        "emulationRealtime": sample.get("emulationRealtime"),
        "novaRealtime": sample.get("novaRealtime"),
        "runId": sample.get("runId"),
        "scenarioId": sample.get("scenarioId"),
        "simulationTime": sample.get("simulationTime"),
    }

    values = sample.get("values", [])

    # In the exported files, values is usually a JSON-encoded string.
    if isinstance(values, str):
        values = json.loads(values)

    for entity_entry in values:
        for _, variables in entity_entry.items():
            row.update(variables)

    return row


def read_data(input_path: Path, filename=None) -> pd.DataFrame:
    """Read a .zst simulation file and return a pandas DataFrame."""

    # If input_path is already a file, ignore filename parameter
    if input_path.is_file():
        files_to_read = [input_path]
    else:
        # Handle filename as list or single value
        if filename is None:
            raise ValueError("filename must be provided when input_path is a directory")

        if isinstance(filename, list):
            files_to_read = [input_path / fn for fn in filename]
        else:
            files_to_read = [input_path / filename]

    all_dataframes = []

    for file_path in files_to_read:
        # Check for alternative file formats
        base_path = file_path.with_suffix("").with_suffix("")  # Remove .json.zst
        json_path = base_path.with_suffix(".json")
        csv_path = base_path.with_suffix(".csv")
        zst_path = Path(str(base_path) + ".json.zst")

        actual_file = None
        file_type = None

        # Priority: .json > .csv > .json.zst
        if json_path.exists():
            actual_file = json_path
            file_type = "json"
            warnings.warn(f"Using {json_path} instead of .json.zst file")
        elif csv_path.exists():
            actual_file = csv_path
            file_type = "csv"
            warnings.warn(f"Using {csv_path} instead of .json.zst file")
        elif zst_path.exists():
            actual_file = zst_path
            file_type = "zst"
        elif file_path.exists():
            actual_file = file_path
            # Determine type from extension
            if file_path.suffix == ".csv":
                file_type = "csv"
            elif file_path.suffix == ".json":
                file_type = "json"
            else:
                file_type = "zst"
        else:
            raise FileNotFoundError(f"No file found for {file_path}")

        # Read the file based on type
        if file_type == "csv":
            df = pd.read_csv(actual_file)
        elif file_type == "json":
            with actual_file.open("r", encoding="utf-8") as f:
                text = f.read()
            df = _process_json_text(text)
        else:  # zst
            text = read_zst_text(actual_file)
            df = _process_json_text(text)

        all_dataframes.append(df)

    # Concatenate all dataframes if multiple files were read
    if len(all_dataframes) == 1:
        return all_dataframes[0]
    else:
        return pd.concat(all_dataframes, ignore_index=True)


def _process_json_text(text: str) -> pd.DataFrame:
    """Process JSON text and return a DataFrame."""
    rows = []
    all_columns = set()

    for sample in iter_json_samples_from_text(text):
        row = flatten_sample(sample)
        rows.append(row)
        all_columns.update(row.keys())

    if not rows:
        return pd.DataFrame()

    metadata_columns = [
        "emulationRealtime",
        "novaRealtime",
        "runId",
        "scenarioId",
        "simulationTime",
    ]

    variable_columns = sorted(c for c in all_columns if c not in metadata_columns)
    columns = metadata_columns + variable_columns

    df = pd.DataFrame(rows, columns=columns)
    return df


if __name__ == "__main__":
    df = read_data(Path("data") / "training" / "run_2026-06-07T154603.json.zst")
    print("Finished reading.")
