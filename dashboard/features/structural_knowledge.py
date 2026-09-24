"""Structural-knowledge-based fault diagnosis feature.

Loads the structural knowledge model (models/structural_knowledge.json,
trained by methods/dx/structural_knowledge.py) together with the range-
monitoring thresholds (models/range_monitoring.json) that flag anomalous
variables, then maps anomalous variables to owning components and their
possible fault types.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from selfx.backend import features
from typing import Any

from dash import Input, Output, dcc, html

import pandas as pd
from scripts.evaluate_methods import extract_fault_labels, diagnosis_events
from .range_monitoring import RangeMonitoring
from .case_based import diagnosis_plot

MODELS_PATH = Path(__file__).resolve().parents[2] / "models"


TABLES = {
    "measurements": "measurements.parquet",
    "parameters": "parameters.parquet",
    "faults": "faults.parquet",
}


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


DATA_PATH = Path(__file__).resolve().parents[2] / "data"


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


def initial_selection(datasets: dict[str, dict[str, Path]]) -> tuple[str | None, str | None]:
    for split_name in split_names(datasets):
        scenario_names = sorted(datasets[split_name], key=scenario_sort_key)
        if scenario_names:
            return split_name, scenario_names[0]
    return None, None


class StructKnowledgeDX(features.Feature):
    """Maps anomalous variables to components and their possible faults."""

    abstract = False
    MODEL_FILENAME = "structural_knowledge.json"

    def icon(self) -> str:
        return "account_tree"

    def _load_model(self, model_path: Path) -> dict:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        thresholds_path = MODELS_PATH / "range_monitoring.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(
                f"{thresholds_path} not found; run methods/ad/range_monitoring.py "
                "first so this feature knows which variables are anomalous."
            )
        model["thresholds"] = json.loads(thresholds_path.read_text(encoding="utf-8"))
        return model

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        import plotly.graph_objects as go

        measurements = load_table(self._datasets, split_name, scenario_name, "measurements")
        if measurements is None:
            return go.Figure(), "No measurements found for this scenario."

        faults = load_table(self._datasets, split_name, scenario_name, "faults")
        if faults is None:
            return go.Figure(), "Fault labels unavailable; diagnosis requires known faulty intervals."
        if not measurements.index.equals(faults.index):
            return go.Figure(), "Measurements and fault labels must have aligned timestamps."
        labels = extract_fault_labels(faults, scenario_name)
        endpoints = sorted({end for end, _ in diagnosis_events(labels)})
        measurements = measurements.iloc[endpoints]
        if measurements.empty:
            return go.Figure(), "No faulty intervals in this scenario."

        violations = pd.DataFrame({
            column: RangeMonitoring._violation_flags(measurements[column], bounds)
            for column, bounds in model["thresholds"].items() if column in measurements
        }, index=measurements.index)
        predictions = []
        for row in violations.itertuples(index=False, name=None):
            hits = {}
            for column, hit in zip(violations.columns, row):
                if hit:
                    component = column.rpartition(".")[0] or column
                    hits[component] = hits.get(component, 0) + 1
            if not hits:
                predictions.append("unknown")
                continue
            component = max(hits, key=hits.get)
            component_type = model["component_type_by_id"].get(component)
            possible = (model["faults_by_component"].get(component, [])
                        if "faults_by_component" in model else model["faults_by_type"].get(component_type, []))
            predictions.append(component + "." + (possible[0] if len(possible) == 1 else "unknown"))
        return diagnosis_plot(
            faults.index, extract_fault_labels(faults, scenario_name),
            pd.Series(predictions, index=measurements.index),
            f"Structural diagnosis - {split_name}/{scenario_name}",
        )

    def __init__(self, tr: Any = None, periodic: bool = False, fetching: bool = False) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._datasets = {}
        if DATA_PATH.exists():
            for split_path in sorted(DATA_PATH.iterdir()):
                if not split_path.is_dir():
                    continue
                scenario_paths = {
                    scenario_path.name: scenario_path
                    for scenario_path in sorted(split_path.iterdir())
                    if scenario_path.is_dir()
                    and any((scenario_path / table_file).exists() for table_file in TABLES.values())
                }
                if scenario_paths:
                    self._datasets[split_path.name] = scenario_paths
        self._model = None
        self._model_error: str | None = None

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _id_prefix(self) -> str:
        system_name = self._css_token(self.plant_name or "system")
        feature_token = self._css_token(self.feature_name().lower())
        return f"{system_name}-{feature_token}"

    def _set_component_ids(self) -> None:
        prefix = self._id_prefix()
        self._file_selector_id = f"{prefix}-file-selector"
        self._graph_id = f"{prefix}-graph"
        self._summary_id = f"{prefix}-summary"

    def _model_path(self) -> Path:
        return MODELS_PATH / self.MODEL_FILENAME

    def _ensure_model(self) -> Any:
        """Load the model artifact once, caching load errors for display."""
        if self._model is None and self._model_error is None:
            model_path = self._model_path()
            if not model_path.exists():
                self._model_error = (
                    f"Model file not found: {model_path}. Run the matching "
                    f"script under methods/ to train it first."
                )
            else:
                try:
                    self._model = self._load_model(model_path)
                except Exception as error:  # noqa: BLE001 - surfaced in the UI
                    self._model_error = f"Could not load model: {error}"
        return self._model

    def topbar_controls(self) -> html.Div | None:
        options = file_options(self._datasets)
        if not options:
            return None
        initial_split, initial_scenario = initial_selection(self._datasets)
        return html.Div(
            [
                dcc.Dropdown(
                    id=self._file_selector_id,
                    options=options,
                    value=file_value(initial_split, initial_scenario)
                    if initial_split
                    else None,
                    clearable=False,
                    className="file_dropdown",
                ),
            ],
            className="topbar_data_controls",
        )

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        return html.Div(
            [
                html.Div(id=self._summary_id),
                dcc.Loading(dcc.Graph(id=self._graph_id)),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

        @dash_app.callback(
            Output(self._graph_id, "figure"),
            Output(self._summary_id, "children"),
            Input(self._file_selector_id, "value"),
        )
        def _update(file_value: str | None):
            import plotly.graph_objects as go

            split_name, scenario_name = split_file(file_value)
            if split_name is None:
                return go.Figure(), "Select a scenario file."

            model = self._ensure_model()
            if model is None:
                return go.Figure(), self._model_error or "Model unavailable."

            try:
                return self._analyze(split_name, scenario_name, model)
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                return go.Figure(), f"Analysis failed: {error}"
