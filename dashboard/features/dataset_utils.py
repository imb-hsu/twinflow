"""Shared helpers for discovering prepared scenario parquet datasets.

Used by DataPlots and the model-based AD/DX features so they list and order
scenario files (training 0pct/1pct/10pct, then test) the same way.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data"
MODELS_PATH = Path(__file__).resolve().parents[2] / "models"
TABLES = {
    "measurements": "measurements.parquet",
    "parameters": "parameters.parquet",
    "faults": "faults.parquet",
}

_SPLIT_ORDER = {"training": 0, "test": 1}


def split_sort_key(split_name: str) -> tuple[int, str]:
    return _SPLIT_ORDER.get(split_name, len(_SPLIT_ORDER)), split_name


def scenario_sort_key(scenario_name: str) -> tuple[float, str]:
    # Scenario names look like "10_0pct_faults_rep_1"; sort by the numeric
    # percentage (0, 1, 10, ...) instead of lexicographically ("10" < "1_").
    match = re.match(r"(\d+)_(\d+)pct", scenario_name)
    if match is None:
        return float("inf"), scenario_name
    return float(f"{match.group(1)}.{match.group(2)}"), scenario_name


def discover_datasets(data_path: Path = DATA_PATH) -> dict[str, dict[str, Path]]:
    if not data_path.exists():
        return {}
    datasets: dict[str, dict[str, Path]] = {}
    for split_path in sorted(data_path.iterdir()):
        if not split_path.is_dir():
            continue
        scenario_paths = {
            scenario_path.name: scenario_path
            for scenario_path in sorted(split_path.iterdir())
            if scenario_path.is_dir()
            and any((scenario_path / table_file).exists() for table_file in TABLES.values())
        }
        if scenario_paths:
            datasets[split_path.name] = scenario_paths
    return datasets


def split_names(datasets: dict[str, dict[str, Path]]) -> tuple[str, ...]:
    return tuple(sorted(datasets, key=split_sort_key))


def file_value(split_name: str, scenario_name: str) -> str:
    return f"{split_name}/{scenario_name}"


def split_file(file_value_str: str | None) -> tuple[str | None, str | None]:
    if not file_value_str or "/" not in file_value_str:
        return None, None
    split_name, scenario_name = file_value_str.split("/", 1)
    return split_name, scenario_name


def file_options(datasets: dict[str, dict[str, Path]]) -> list[dict[str, str]]:
    return [
        {
            "label": f"{split_name}/{scenario_name}.json",
            "value": file_value(split_name, scenario_name),
        }
        for split_name in split_names(datasets)
        for scenario_name in sorted(datasets[split_name], key=scenario_sort_key)
    ]


def scenario_path(
    datasets: dict[str, dict[str, Path]], split_name: str | None, scenario_name: str | None
) -> Path | None:
    if split_name is None or scenario_name is None:
        return None
    return datasets.get(split_name, {}).get(scenario_name)


def load_table(
    datasets: dict[str, dict[str, Path]],
    split_name: str | None,
    scenario_name: str | None,
    table: str,
) -> pd.DataFrame | None:
    path = scenario_path(datasets, split_name, scenario_name)
    if path is None:
        return None
    file_path = path / TABLES[table]
    if not file_path.exists():
        return None
    return pd.read_parquet(file_path)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column].dtype)
        and not pd.api.types.is_bool_dtype(df[column].dtype)
    ]


def initial_selection(datasets: dict[str, dict[str, Path]]) -> tuple[str | None, str | None]:
    for split_name in split_names(datasets):
        scenario_names = sorted(datasets[split_name], key=scenario_sort_key)
        if scenario_names:
            return split_name, scenario_names[0]
    return None, None
