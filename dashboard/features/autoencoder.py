"""Autoencoder-based anomaly detection feature.

Loads the trained scikit-learn autoencoder (models/autoencoder.json, trained
by methods/ad/autoencoder.py) and plots its per-sample reconstruction error
against the learned threshold for the selected scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from selfx.backend import features
from typing import Any

import numpy as np
from dash import Input, Output, dcc, html

import pandas as pd
from plotly.subplots import make_subplots

from .range_monitoring import anomaly_stateflow, plot_stateflow

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


MODELS_PATH = Path(__file__).resolve().parents[2] / "models"


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


def encode_measurements(measurements: pd.DataFrame, model: dict) -> np.ndarray:
    """Apply saved feature order and one-hot categories; unseen categories are all-zero."""
    columns = model["columns"]
    missing = set(columns) - set(measurements.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    if "encoding" not in model:
        # Existing numeric-only JSON models remain usable until retrained.
        fill_values = dict(zip(columns, model["scaler"]["mean"]))
        return measurements[columns].fillna(value=fill_values).to_numpy(dtype=float)
    encoded = []
    for column in columns:
        spec = model["encoding"][column]
        values = measurements[column]
        if spec["type"] == "boolean":
            encoded.append(values.fillna(False).to_numpy(dtype=float)[:, None])
        elif spec["type"] == "numeric":
            encoded.append(values.fillna(spec["fill_value"]).to_numpy(dtype=float)[:, None])
        else:
            values = values.astype("string")
            encoded.extend(values.eq(category).fillna(False).to_numpy(dtype=float)[:, None]
                           for category in spec["categories"])
    return np.concatenate(encoded, axis=1)


class AutoencoderAD(features.Feature):
    """Anomaly score based on autoencoder reconstruction error."""

    abstract = False
    MODEL_FILENAME = "autoencoder.json"

    def icon(self) -> str:
        return "psychology"

    def _load_model(self, model_path: Path) -> dict:
        return json.loads(model_path.read_text(encoding="utf-8"))

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        import plotly.graph_objects as go

        measurements = load_table(self._datasets, split_name, scenario_name, "measurements")
        if measurements is None:
            return go.Figure(), "No measurements found for this scenario."

        columns = model["columns"]
        missing = [column for column in columns if column not in measurements.columns]
        if missing:
            return (
                go.Figure(),
                f"Scenario is missing {len(missing)} of {len(columns)} model columns "
                f"(e.g. {missing[0]}).",
            )

        X = encode_measurements(measurements, model)
        X_scaled = (X - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
        reconstructed = X_scaled
        layers = model["layers"]
        for index, layer in enumerate(layers):
            reconstructed = reconstructed @ np.asarray(layer["weights"]) + np.asarray(layer["bias"])
            if index < len(layers) - 1:
                reconstructed = np.maximum(reconstructed, 0)
        reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        threshold = model["threshold"]

        anomalous = reconstruction_error > threshold
        faults = load_table(self._datasets, split_name, scenario_name, "faults")
        stateflow = anomaly_stateflow(measurements.index, anomalous, faults)
        colors = {state: "#dc2626" for state in stateflow.State.unique()}
        colors.update({"Nominal": "#0d9488", "Unavailable": "#94a3b8"})
        timeline = plot_stateflow(
            stateflow, state_col="State", task_col="Task", start_column="Start",
            finish_column="Finish", return_figure=True, description_col=None,
            color_mapping=colors, bar_height=20,
        )
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.3, 0.7], vertical_spacing=0.09)
        for trace in timeline.data:
            if trace.name not in {"Nominal", "Anomaly", "Unavailable"}:
                trace.showlegend = False
            if trace.hoverinfo == "skip":
                trace.update(hoverinfo="text", hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{y}<br>Time: %{x}s<extra></extra>"
                ))
            else:
                trace.text = [trace.name if task == "Target" and trace.name not in
                              {"Nominal", "Unavailable"} else "" for task in trace.y]
                trace.textposition = "top center"
                trace.textfont = dict(size=11, color="#191835")
            figure.add_trace(trace, row=1, col=1)
        figure.add_trace(
            go.Scatter(
                x=measurements.index,
                y=reconstruction_error,
                mode="lines",
                name="Reconstruction error",
            ), row=2, col=1,
        )
        figure.add_hline(y=threshold, line_dash="dash", line_color="red",
                         annotation_text="threshold", row=2, col=1)
        figure.update_yaxes(type="category", categoryorder="array",
                            categoryarray=["Target", "Predicted"], row=1, col=1)
        figure.update_yaxes(title_text="Reconstruction error", row=2, col=1)
        figure.update_xaxes(title_text="simulationTime [s]", row=2, col=1)
        figure.update_layout(
            title=f"Autoencoder anomaly score — {split_name}/{scenario_name}",
            template="plotly_white", height=760, hovermode="x unified",
            legend=dict(orientation="h", y=-0.12),
            margin=dict(l=85, r=30, t=60, b=90),
        )

        summary = html.Div(
            html.P(
                f"{int(anomalous.sum())}/{len(measurements)} samples exceed the "
                f"reconstruction-error threshold ({threshold:.4f})."
            )
        )
        return figure, summary

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
