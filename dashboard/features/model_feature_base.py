"""Shared layout/callback scaffolding for model-based AD/DX dashboard features.

Each subclass loads a model artifact (trained by a script under scripts/) and
applies it to the scenario parquet data currently selected in its file dropdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import Input, Output, dcc, html
from selfx.backend import features

from . import dataset_utils as ds


class ModelFeatureBase(features.Feature):
    """Base class rendering a file selector, a graph, and a summary panel."""

    abstract = True
    # Subclasses set this to the artifact filename under dataset_utils.MODELS_PATH.
    MODEL_FILENAME: str = ""

    def __init__(self, tr: Any = None, periodic: bool = False, fetching: bool = False) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._datasets = ds.discover_datasets()
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
        return ds.MODELS_PATH / self.MODEL_FILENAME

    def _ensure_model(self) -> Any:
        """Load the model artifact once, caching load errors for display."""
        if self._model is None and self._model_error is None:
            model_path = self._model_path()
            if not model_path.exists():
                self._model_error = (
                    f"Model file not found: {model_path}. Run the matching "
                    f"script under scripts/ to train it first."
                )
            else:
                try:
                    self._model = self._load_model(model_path)
                except Exception as error:  # noqa: BLE001 - surfaced in the UI
                    self._model_error = f"Could not load model: {error}"
        return self._model

    def _load_model(self, model_path: Path) -> Any:
        raise NotImplementedError

    def _analyze(self, split_name: str, scenario_name: str, model: Any):
        """Return (plotly Figure, summary children) for the selected scenario."""
        raise NotImplementedError

    def topbar_controls(self) -> html.Div | None:
        options = ds.file_options(self._datasets)
        if not options:
            return None
        initial_split, initial_scenario = ds.initial_selection(self._datasets)
        return html.Div(
            [
                dcc.Dropdown(
                    id=self._file_selector_id,
                    options=options,
                    value=ds.file_value(initial_split, initial_scenario)
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

            split_name, scenario_name = ds.split_file(file_value)
            if split_name is None:
                return go.Figure(), "Select a scenario file."

            model = self._ensure_model()
            if model is None:
                return go.Figure(), self._model_error or "Model unavailable."

            try:
                return self._analyze(split_name, scenario_name, model)
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                return go.Figure(), f"Analysis failed: {error}"
