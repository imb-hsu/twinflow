"""Interactive SelfX view for plotting prepared TwinFlow scenario data."""

from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from dash import Input, Output, dcc, html
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from selfx.backend import features

try:
    from ml4cps import plot_stateflow
except ImportError:
    from ml4cps.vis import plot_stateflow


DATA_PATH = Path(__file__).resolve().parents[2] / "data"
SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "scenarios"
MAX_DEFAULT_COLUMNS = 3
MAX_PLOT_POINTS = 5_000
SCENARIO_EVENT_DURATION_SECONDS = 20
TABLES = {
    "measurements": "measurements.parquet",
    "parameters": "parameters.parquet",
    "faults": "faults.parquet",
}
DEFAULT_TABLE = "measurements"


class DataPlots(features.Feature):
    """Render selectable line plots for prepared scenario parquet data."""

    abstract = False

    def __init__(
        self,
        tr: Any = None,
        periodic: bool = False,
        fetching: bool = False,
    ) -> None:
        super().__init__(tr=tr, periodic=periodic, fetching=fetching)
        self._callbacks_registered = False
        self._set_component_ids()
        self._datasets = self._discover_datasets()
        self._validate_datasets()
        self._split_names = tuple(sorted(self._datasets))

    @staticmethod
    def _css_token(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in value
        )

    def _set_component_ids(self) -> None:
        system_name = self._css_token(self.plant_name or "system")
        prefix = f"{system_name}-data-plots"
        self._file_selector_id = f"{prefix}-file-selector"
        self._columns_selector_id = f"{prefix}-columns-selector"
        self._graph_id = f"{prefix}-graph"
        self._scenario_info_id = f"{prefix}-scenario-info"
        self._summary_id = f"{prefix}-summary"

    @staticmethod
    def _discover_datasets() -> dict[str, dict[str, Path]]:
        if not DATA_PATH.exists():
            return {}

        datasets: dict[str, dict[str, Path]] = {}
        for split_path in sorted(DATA_PATH.iterdir()):
            if not split_path.is_dir():
                continue

            scenario_paths = {
                scenario_path.name: scenario_path
                for scenario_path in sorted(split_path.iterdir())
                if scenario_path.is_dir()
                and any(
                    (scenario_path / table_file).exists()
                    for table_file in TABLES.values()
                )
            }
            if scenario_paths:
                datasets[split_path.name] = scenario_paths

        return datasets

    def _validate_datasets(self) -> None:
        if not self._datasets:
            raise ValueError("No prepared scenario parquet datasets found in data/")

    def _scenario_path(self, split_name: str, scenario_name: str) -> Path | None:
        return self._datasets.get(split_name, {}).get(scenario_name)

    def _file_options(self) -> list[dict[str, str]]:
        return [
            {
                "label": f"{split_name}/{scenario_name}.json",
                "value": self._file_value(split_name, scenario_name),
            }
            for split_name in self._split_names
            for scenario_name in sorted(self._datasets[split_name])
        ]

    @staticmethod
    def _file_value(split_name: str, scenario_name: str) -> str:
        return f"{split_name}/{scenario_name}"

    @staticmethod
    def _split_file(file_value: str | None) -> tuple[str | None, str | None]:
        if not file_value or "/" not in file_value:
            return None, None
        split_name, scenario_name = file_value.split("/", 1)
        return split_name, scenario_name

    def _available_tables(
        self,
        split_name: str | None,
        scenario_name: str | None,
    ) -> tuple[str, ...]:
        scenario_path = self._scenario_path(split_name, scenario_name)
        if scenario_path is None:
            return ()

        return tuple(
            table_name
            for table_name, table_file in TABLES.items()
            if (scenario_path / table_file).exists()
        )

    def _table_path(
        self,
        split_name: str | None,
        scenario_name: str | None,
        table_name: str | None,
    ) -> Path | None:
        scenario_path = self._scenario_path(split_name, scenario_name)
        table_file = TABLES.get(table_name)
        if scenario_path is None or table_file is None:
            return None

        table_path = scenario_path / table_file
        if not table_path.exists():
            return None
        return table_path

    @staticmethod
    def _scenario_definition_path(
        split_name: str | None,
        scenario_name: str | None,
    ) -> Path | None:
        if not split_name or not scenario_name:
            return None

        scenario_path = SCENARIOS_PATH / split_name / f"{scenario_name}.json"
        if scenario_path.exists():
            return scenario_path
        return None

    @staticmethod
    def _read_scenario_definition(
        split_name: str | None,
        scenario_name: str | None,
    ) -> dict[str, Any] | None:
        scenario_path = DataPlots._scenario_definition_path(split_name, scenario_name)
        if scenario_path is None:
            return None

        try:
            return json.loads(scenario_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None

    def _columns(
        self,
        split_name: str | None,
        scenario_name: str | None,
        table_name: str | None,
    ) -> tuple[str, ...]:
        table_path = self._table_path(split_name, scenario_name, table_name)
        if table_path is None:
            return ()

        try:
            dataframe = pd.read_parquet(table_path)
        except Exception:
            return ()

        return tuple(str(column) for column in dataframe.columns)

    def _read_plot_data(
        self,
        split_name: str | None,
        scenario_name: str | None,
        table_name: str | None,
        columns: list[str],
    ) -> pd.DataFrame:
        table_path = self._table_path(split_name, scenario_name, table_name)
        if table_path is None:
            raise ValueError("Selected data table does not exist")

        dataframe = pd.read_parquet(table_path, columns=columns)
        if len(dataframe) > MAX_PLOT_POINTS:
            step = math.ceil(len(dataframe) / MAX_PLOT_POINTS)
            dataframe = dataframe.iloc[::step]
        return dataframe

    @staticmethod
    def _empty_figure(message: str) -> dict[str, Any]:
        return {
            "data": [],
            "layout": {
                "annotations": [
                    {
                        "showarrow": False,
                        "text": message,
                        "x": 0.5,
                        "xref": "paper",
                        "y": 0.5,
                        "yref": "paper",
                    }
                ],
                "height": 520,
                "margin": {"b": 48, "l": 56, "r": 24, "t": 40},
                "template": "plotly_white",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }

    @staticmethod
    def _dropdown_options(values: tuple[str, ...]) -> list[dict[str, str]]:
        return [{"label": value, "value": value} for value in values]

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _parse_scenario_time(value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return int(value)

        parts = str(value).strip().split(":")
        if len(parts) == 2:
            try:
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            except ValueError:
                return None

        if len(parts) == 3:
            try:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            except ValueError:
                return None

        return None

    @staticmethod
    def _is_reset_value(value: Any) -> bool:
        if isinstance(value, bool):
            return not value
        if value is None:
            return True

        normalized_value = str(value).strip().lower().replace(",", ".")
        if normalized_value in {"", "false", "off", "none", "null"}:
            return True

        try:
            return float(normalized_value) == 0
        except ValueError:
            return False

    @staticmethod
    def _is_pallet_creator_action(action: dict[str, Any]) -> bool:
        element = DataPlots._format_value(action.get("element")).lower()
        property_name = DataPlots._format_value(action.get("property")).lower()
        return element == "palletcreator" or property_name == "createpallet"

    @staticmethod
    def _metadata_field(label: str, value: Any) -> html.Div:
        return html.Div(
            [
                html.Div(label, style={"color": "#64748b", "fontSize": "0.78rem"}),
                html.Div(
                    DataPlots._format_value(value),
                    style={
                        "color": "#0f172a",
                        "fontSize": "0.92rem",
                        "fontWeight": 600,
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    },
                    title=DataPlots._format_value(value),
                ),
            ],
            style={
                "backgroundColor": "#ffffff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "0.45rem",
                "minWidth": "9rem",
                "padding": "0.55rem 0.7rem",
            },
        )

    def _scenario_stateflow_dataframe(
        self,
        scenario_definition: dict[str, Any],
    ) -> pd.DataFrame:
        raw_actions = scenario_definition.get("actions") or []
        actions = [action for action in raw_actions if isinstance(action, dict)]
        if not actions:
            return pd.DataFrame(columns=["Task", "State", "Start", "Finish"])

        scenario = scenario_definition.get("scenario") or {}
        scenario_duration = self._parse_scenario_time(scenario.get("duration"))
        parsed_actions: list[dict[str, Any]] = []
        for order, action in enumerate(actions):
            if self._is_pallet_creator_action(action):
                continue

            start = self._parse_scenario_time(action.get("startsAt"))
            if start is None:
                continue

            parsed_actions.append(
                {
                    "order": order,
                    "start": start,
                    "area": self._format_value(action.get("area")) or "Scenario",
                    "element": self._format_value(action.get("element")) or "Action",
                    "property": self._format_value(action.get("property")) or "Action",
                    "value": action.get("value"),
                }
            )

        if not parsed_actions:
            return pd.DataFrame(columns=["Task", "State", "Start", "Finish"])

        parsed_actions.sort(key=lambda action: (action["start"], action["order"]))
        fallback_finish = max(
            scenario_duration or 0,
            max(action["start"] for action in parsed_actions)
            + SCENARIO_EVENT_DURATION_SECONDS,
        )

        rows: list[dict[str, Any]] = []
        resettable_keys = {
            (action["area"], action["element"], action["property"])
            for action in parsed_actions
            if self._is_reset_value(action["value"])
        }
        active_actions: dict[tuple[str, str, str], dict[str, Any]] = {}

        def append_interval(
            action: dict[str, Any],
            finish: int,
        ) -> None:
            start = action["start"]
            if finish <= start:
                finish = start + SCENARIO_EVENT_DURATION_SECONDS

            rows.append(
                {
                    "Task": f"{action['area']} / {action['element']}",
                    "State": action["property"],
                    "Start": start,
                    "Finish": finish,
                    "Area": action["area"],
                    "Element": action["element"],
                    "Value": self._format_value(action["value"]),
                    "Description": (
                        f"{action['property']} = "
                        f"{self._format_value(action['value'])}"
                    ),
                }
            )

        for action in parsed_actions:
            key = (action["area"], action["element"], action["property"])
            if key not in resettable_keys:
                append_interval(
                    action,
                    min(
                        fallback_finish,
                        action["start"] + SCENARIO_EVENT_DURATION_SECONDS,
                    ),
                )
                continue

            active_action = active_actions.pop(key, None)

            if self._is_reset_value(action["value"]):
                if active_action is not None:
                    append_interval(active_action, action["start"])
                continue

            if active_action is not None:
                append_interval(active_action, action["start"])

            active_actions[key] = action

        for active_action in active_actions.values():
            append_interval(active_action, fallback_finish)

        if not rows:
            return pd.DataFrame(columns=["Task", "State", "Start", "Finish"])

        return pd.DataFrame(rows).sort_values(["Start", "Task", "State"])

    def _pallet_creator_events_dataframe(
        self,
        scenario_definition: dict[str, Any],
    ) -> pd.DataFrame:
        raw_actions = scenario_definition.get("actions") or []
        rows: list[dict[str, Any]] = []
        for action in raw_actions:
            if not isinstance(action, dict):
                continue
            if not self._is_pallet_creator_action(action):
                continue

            start = self._parse_scenario_time(action.get("startsAt"))
            if start is None:
                continue

            rows.append(
                {
                    "Time": start,
                    "Event": "CreatePallet",
                    "Area": self._format_value(action.get("area")),
                    "Element": self._format_value(action.get("element")),
                    "Value": self._format_value(action.get("value")),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["Time", "Event", "Area", "Element", "Value"])

        return pd.DataFrame(rows).sort_values(["Time", "Event"])

    def _scenario_information(
        self,
        split_name: str | None,
        scenario_name: str | None,
    ) -> html.Div:
        scenario_definition = self._read_scenario_definition(split_name, scenario_name)
        if scenario_definition is None:
            return html.Div(
                "No scenario definition JSON was found for this selected scenario.",
                style={"color": "#64748b"},
            )

        scenario = scenario_definition.get("scenario") or {}
        actions = scenario_definition.get("actions") or []
        if not isinstance(actions, list):
            actions = []

        action_type_counts = Counter(
            str(action.get("property") or "Action")
            for action in actions
            if isinstance(action, dict)
        )
        metadata_fields = [
            ("Split", split_name),
            ("Scenario", scenario.get("id") or scenario_name),
            ("Duration", scenario.get("duration")),
            ("Cycles", scenario.get("cycles")),
            ("Start", scenario.get("startDate")),
        ]
        metadata_fields.extend(
            (action_type, count)
            for action_type, count in sorted(
                action_type_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

        return html.Div(
            [
                html.Div(
                    "Scenario information",
                    style={
                        "color": "#0f172a",
                        "fontSize": "1rem",
                        "fontWeight": 700,
                        "marginBottom": "0.75rem",
                    },
                ),
                html.Div(
                    [
                        self._metadata_field(label, value)
                        for label, value in metadata_fields
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "0.6rem",
                        "marginBottom": "0.9rem",
                    },
                ),
            ],
            style={
                "backgroundColor": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "borderRadius": "0.5rem",
                "marginTop": "1.5rem",
                "padding": "1rem",
            },
        )

    @staticmethod
    def _json_values(values: Any) -> list[Any]:
        return [None if pd.isna(value) else value for value in values]

    @staticmethod
    def _add_empty_row_message(
        figure: Any,
        row: int,
        message: str,
    ) -> None:
        figure.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="text",
                text=[message],
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    def _plot_figure(
        self,
        split_name: str,
        scenario_name: str,
        table_name: str,
        columns: list[str] | None,
    ) -> tuple[Any, html.Div]:
        selected_columns = [column for column in columns or [] if column]
        scenario_definition = self._read_scenario_definition(split_name, scenario_name)
        stateflow = (
            self._scenario_stateflow_dataframe(scenario_definition)
            if scenario_definition is not None
            else pd.DataFrame(columns=["Task", "State", "Start", "Finish"])
        )
        pallet_events = (
            self._pallet_creator_events_dataframe(scenario_definition)
            if scenario_definition is not None
            else pd.DataFrame(columns=["Time", "Event", "Area", "Element", "Value"])
        )
        data_load_error: Exception | None = None
        dataframe = pd.DataFrame()

        if selected_columns:
            try:
                dataframe = self._read_plot_data(
                    split_name,
                    scenario_name,
                    table_name,
                    selected_columns,
                )
            except Exception as error:
                data_load_error = error

        stateflow_task_count = (
            len(stateflow["Task"].unique())
            if "Task" in stateflow and not stateflow.empty
            else 1
        )
        figure = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.055,
            row_heights=[0.26, 0.12, 0.62],
            subplot_titles=[
                "Scenario stateflow",
                "PalletCreator events",
                f"{TABLES.get(table_name, table_name)} data",
            ],
        )

        if stateflow.empty:
            self._add_empty_row_message(figure, 1, "No scenario stateflow actions")
        else:
            scenario_figure = plot_stateflow(
                stateflow,
                state_col="State",
                task_col="Task",
                start_column="Start",
                finish_column="Finish",
                return_figure=True,
                description_col=None,
            )
            for trace in scenario_figure.data:
                if getattr(trace, "hoverinfo", None) == "skip":
                    trace.update(
                        hoverinfo="text",
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Time: %{x}s<br>"
                            "Task: %{y}"
                            "<extra></extra>"
                        ),
                    )
                figure.add_trace(trace, row=1, col=1)

        if pallet_events.empty:
            self._add_empty_row_message(figure, 2, "No PalletCreator events")
        else:
            figure.add_trace(
                go.Scatter(
                    x=self._json_values(pallet_events["Time"]),
                    y=["CreatePallet"] * len(pallet_events),
                    mode="markers",
                    marker={
                        "color": "#7c3aed",
                        "line": {"color": "#4c1d95", "width": 1},
                        "size": 9,
                        "symbol": "diamond",
                    },
                    name="CreatePallet",
                    customdata=pallet_events[["Area", "Element", "Value"]].values,
                    hovertemplate=(
                        "<b>CreatePallet</b><br>"
                        "Time: %{x}s<br>"
                        "Area: %{customdata[0]}<br>"
                        "Element: %{customdata[1]}<br>"
                        "Value: %{customdata[2]}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        data_traces = []
        if data_load_error is not None:
            self._add_empty_row_message(
                figure,
                3,
                f"Could not load data: {data_load_error}",
            )
        elif not selected_columns:
            self._add_empty_row_message(figure, 3, "Select at least one column")
        else:
            data_traces = [
                go.Scatter(
                    mode="lines",
                    name=column,
                    x=self._json_values(dataframe.index),
                    y=self._json_values(dataframe[column]),
                    hovertemplate=(
                        f"<b>{column}</b><br>"
                        "Time: %{x}s<br>"
                        "Value: %{y}"
                        "<extra></extra>"
                    ),
                )
                for column in selected_columns
                if column in dataframe
            ]
            if not data_traces:
                self._add_empty_row_message(
                    figure,
                    3,
                    "Selected columns were not found",
                )
            for trace in data_traces:
                figure.add_trace(trace, row=3, col=1)

        figure.update_layout(
            height=max(760, min(1200, 680 + stateflow_task_count * 18)),
            hovermode="closest",
            hoverlabel={"align": "left"},
            legend={"orientation": "h", "y": -0.18},
            margin={"b": 112, "l": 96, "r": 24, "t": 72},
            template="plotly_white",
            title=f"{scenario_name} / {table_name}",
        )
        figure.update_xaxes(matches="x")
        figure.update_xaxes(title_text="simulationTime [s]", row=3, col=1)
        figure.update_yaxes(title_text="scenario", row=1, col=1)
        figure.update_yaxes(title_text="events", row=2, col=1)
        figure.update_yaxes(title_text="value", row=3, col=1)

        if data_load_error is not None:
            return (
                figure,
                html.Div(str(data_load_error), style={"color": "#b91c1c"}),
            )

        summary = html.Div(
            [
                html.Span(f"{len(dataframe)} samples" if selected_columns else "0 samples"),
                html.Span(" | "),
                html.Span(f"{len(data_traces)} plotted columns"),
                html.Span(" | "),
                html.Span(f"{len(stateflow)} scenario intervals"),
                html.Span(" | "),
                html.Span(f"{len(pallet_events)} PalletCreator events"),
            ],
            style={"color": "#475569", "fontSize": "0.9rem"},
        )
        return figure, summary

    def _initial_selection(self) -> tuple[str, str, str]:
        split_name = self._split_names[0]
        scenario_name = sorted(self._datasets[split_name])[0]
        return split_name, scenario_name, self._file_value(split_name, scenario_name)

    def perform(self, start: Any, end: Any) -> dict[str, int]:
        """Return static metadata when SelfX requests feature computation."""
        scenario_count = sum(len(scenarios) for scenarios in self._datasets.values())
        table_count = sum(
            len(self._available_tables(split_name, scenario_name))
            for split_name, scenarios in self._datasets.items()
            for scenario_name in scenarios
        )
        return {
            "splits": len(self._datasets),
            "scenarios": scenario_count,
            "tables": table_count,
        }

    def is_online(self, role: Any) -> bool:
        """Render static prepared data without time-series analysis."""
        return True

    def icon(self) -> str:
        """Return the Material icon for this feature."""
        return "show_chart"

    def topbar_controls(self) -> html.Div:
        """Build the scenario file selector for the SelfX topbar."""
        self._set_component_ids()
        _, _, initial_file = self._initial_selection()
        return html.Div(
            [
                dcc.Dropdown(
                    id=self._file_selector_id,
                    options=self._file_options(),
                    value=initial_file,
                    clearable=False,
                    className="file_dropdown",
                ),
            ],
            className="topbar_data_controls",
        )

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        """Build the data plotting layout."""
        initial_split, initial_scenario, _ = self._initial_selection()
        initial_columns = self._columns(
            initial_split,
            initial_scenario,
            DEFAULT_TABLE,
        )
        initial_selected_columns = list(initial_columns[:MAX_DEFAULT_COLUMNS])
        initial_figure, initial_summary = self._plot_figure(
            initial_split,
            initial_scenario,
            DEFAULT_TABLE,
            initial_selected_columns,
        )
        initial_scenario_information = self._scenario_information(
            initial_split,
            initial_scenario,
        )
        return html.Div(
            [
                html.Div(
                    id=self._scenario_info_id,
                    children=initial_scenario_information,
                ),
                html.Div(
                    [
                        html.Label("Selected Measurements"),
                        dcc.Dropdown(
                            id=self._columns_selector_id,
                            options=self._dropdown_options(initial_columns),
                            value=initial_selected_columns,
                            clearable=False,
                            multi=True,
                            className="control_very_wide",
                        ),
                    ],
                    className="content_control_wide control_very_wide_container",
                ),
                html.Div(
                    [
                        dcc.Graph(id=self._graph_id, figure=initial_figure),
                        html.Div(id=self._summary_id, children=initial_summary),
                    ],
                    style={
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "0.5rem",
                        "marginTop": "1.5rem",
                        "padding": "1rem",
                    },
                ),
            ],
            style={"padding": "1rem"},
        )

    def register_callbacks(self, dash_app: Any, analysis: Any) -> None:
        """Register data selection and plotting callbacks."""
        if self._callbacks_registered:
            return

        self._set_component_ids()

        @dash_app.callback(
            Output(self._columns_selector_id, "options"),
            Output(self._columns_selector_id, "value"),
            Input(self._file_selector_id, "value"),
        )
        def update_columns(
            file_value: str,
        ) -> tuple[list[dict[str, str]], list[str]]:
            split_name, scenario_name = self._split_file(file_value)
            column_names = self._columns(split_name, scenario_name, DEFAULT_TABLE)
            selected_columns = list(column_names[:MAX_DEFAULT_COLUMNS])
            return self._dropdown_options(column_names), selected_columns

        @dash_app.callback(
            Output(self._scenario_info_id, "children"),
            Input(self._file_selector_id, "value"),
        )
        def update_scenario_information(
            file_value: str,
        ) -> html.Div:
            split_name, scenario_name = self._split_file(file_value)
            return self._scenario_information(split_name, scenario_name)

        @dash_app.callback(
            Output(self._graph_id, "figure"),
            Output(self._summary_id, "children"),
            Input(self._file_selector_id, "value"),
            Input(self._columns_selector_id, "value"),
        )
        def update_plot(
            file_value: str,
            columns: list[str] | None,
        ) -> tuple[Any, html.Div]:
            split_name, scenario_name = self._split_file(file_value)
            return self._plot_figure(split_name, scenario_name, DEFAULT_TABLE, columns)

        self._callbacks_registered = True
