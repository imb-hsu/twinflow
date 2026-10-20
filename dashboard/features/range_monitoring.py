"""Range-monitoring anomaly detection feature.

Loads numeric bounds and categorical values (models/range_monitoring.json, trained by
methods/ad/range_monitoring.py) and flags out-of-range samples in the
selected scenario's measurements.

The overview compares predicted and target anomaly states; the signal inspector shows the raw
complete signal and training bounds, with anomalies marked in red.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from selfx.backend import features
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Input, Output, dcc, html
from methods.ad._signals import is_ad_signal, missing_value_warning
from methods.ad.range_monitoring import predict_column_anomalies, violation_flags

try:
    from ml4cps import plot_stateflow
except ImportError:
    from ml4cps.vis import plot_stateflow

TABLES = {
    "measurements": "measurements.parquet",
    "parameters": "parameters.parquet",
    "faults": "faults.parquet",
}


def split_file(file_value_str: str | None) -> tuple[str | None, str | None]:
    if not file_value_str or "/" not in file_value_str:
        return None, None
    split_name, scenario_name = file_value_str.split("/", 1)
    return split_name, scenario_name


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


def _empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(template="plotly_white", height=350)
    return figure


def _state_intervals(index, states, task):
    """Represent every state run, including one-sample anomalies, as an interval."""
    times = np.asarray(index, dtype=float)
    states = np.asarray(states)
    if not len(times):
        return []
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    # Hold each sampled state until the next timestamp. The final sample uses
    # the median sampling interval; a singleton stays visible as a marker.
    finish = times[-1] + (float(np.median(np.diff(times))) if len(times) > 1 else 0.0)
    ends = np.r_[times[starts[1:]], finish]
    return [{"Task": task, "State": states[start], "Start": times[start], "Finish": end}
            for start, end in zip(starts, ends)]


def _target_labels(index, faults):
    """Keep exact component.fault labels, including simultaneous faults."""
    target_states = np.full(len(index), "Unavailable", dtype=object)
    active = np.zeros(len(index), dtype=bool)
    known = np.zeros(len(index), dtype=bool)
    if faults is not None:
        columns = [column for column in faults
                   if (pd.api.types.is_bool_dtype(faults[column].dtype)
                       or pd.api.types.is_numeric_dtype(faults[column].dtype))
                   and faults[column].notna().any()]
        if columns:
            aligned = faults[columns].reindex(index)
            labels = aligned.notna() & aligned.ne(0).fillna(False)
            # The prepared recordings use 0.1 as the nominal damping baseline.
            # Both reset (0) and that baseline must remain nominal.
            for column in columns:
                if str(column).rsplit(".", 1)[-1] == "IncreasedDampingFault":
                    labels[column] &= aligned[column].ne(0.1).fillna(False)
            active = labels.any(axis=1).to_numpy(dtype=bool)
            known = aligned.notna().all(axis=1).to_numpy(dtype=bool)
            target_states[:] = "Nominal"
            for column in columns:
                mask = labels[column].to_numpy(dtype=bool)
                target_states[mask] = np.where(
                    target_states[mask] == "Nominal", column,
                    target_states[mask] + " + " + column,
                )
            target_states[~known & ~active] = "Unavailable"
    # A known active fault proves a positive target even if another label is missing.
    return target_states, active, known | active


def anomaly_stateflow(index, predicted, faults):
    rows = _state_intervals(index, np.where(predicted, "Anomaly", "Nominal"), "Predicted")
    target_states, _, _ = _target_labels(index, faults)
    rows.extend(_state_intervals(index, target_states, "Target"))
    return pd.DataFrame(rows, columns=["Task", "State", "Start", "Finish"])


class RangeMonitoring(features.Feature):
    """Anomaly score based on per-column min/max range violations."""

    abstract = False
    MODEL_FILENAME = "range_monitoring.json"
    TRAINING_SCRIPT = "range_monitoring.py"

    def _set_component_ids(self):
        prefix = self._id_prefix()
        self._file_selector_id = f"{prefix}-file-selector"
        self._graph_id = f"{prefix}-graph"
        self._summary_id = f"{prefix}-summary"
        prefix = self._id_prefix()
        self._signal_id = f"{prefix}-signal"
        self._all_signal_id = f"{prefix}-all-signal"
        self._overview_store_id = f"{prefix}-overview"

    def icon(self) -> str:
        return "rule"

    def _load_model(self, model_path: Path) -> dict:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(model, dict) or not model:
            raise ValueError("The range-monitoring model must contain signal thresholds.")
        for column, bounds in model.items():
            if not is_ad_signal(column):
                continue
            if "state_column" in bounds:
                if not isinstance(bounds.get("states"), dict) or not bounds["states"]:
                    raise ValueError(f"Missing state profiles for {column}")
                profiles = bounds["states"].values()
            else:
                profiles = [bounds]
            for profile in profiles:
                if profile.get("type", "numeric") == "numeric":
                    profile["min"] = profile.get("lower", profile.get("min"))
                    profile["max"] = profile.get("upper", profile.get("max"))
                    if profile["min"] is None or profile["max"] is None:
                        raise ValueError(f"Missing numeric bounds for {column}")
                elif profile["type"] == "categorical":
                    if not isinstance(profile.get("categories"), list):
                        raise ValueError(f"Missing categories for {column}")
                else:
                    raise ValueError(f"Unknown threshold type for {column}")
        return {column: bounds for column, bounds in model.items() if is_ad_signal(column)}

    @staticmethod
    def _violation_flags(series, bounds):
        return violation_flags(series, bounds)

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        measurements = load_table(self._datasets, split_name, scenario_name, "measurements")
        faults = load_table(self._datasets, split_name, scenario_name, "faults")
        figure, summary, _ = self._overview(measurements, model, f"{split_name}/{scenario_name}", faults)
        return figure, summary

    def _overview(self, measurements, thresholds, title, faults=None):
        if measurements is None:
            return _empty_figure("No measurements found."), "No measurements found for this scenario.", []

        monitored_columns = [column for column in measurements.columns
                             if column in thresholds and is_ad_signal(column)]
        if not monitored_columns:
            return _empty_figure("No monitored signals found."), "None of the trained columns are present in this scenario.", []

        measurements = measurements.sort_index()
        out_of_range = predict_column_anomalies(
            measurements, {column: thresholds[column] for column in monitored_columns}, warn=False
        )
        predicted = out_of_range.any(axis=1)
        stateflow = anomaly_stateflow(measurements.index, predicted, faults)
        colors = {state: "#dc2626" for state in stateflow.State.unique()}
        colors.update({"Nominal": "#0d9488", "Unavailable": "#94a3b8"})
        figure = plot_stateflow(
            stateflow, state_col="State", task_col="Task", start_column="Start",
            finish_column="Finish", return_figure=True, description_col=None,
            color_mapping=colors,
            bar_height=20,
        )
        for trace in figure.data:
            if trace.name not in {"Nominal", "Anomaly", "Unavailable"}:
                trace.showlegend = False
            if trace.hoverinfo == "skip":
                trace.update(hoverinfo="text", hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{y}<br>Time: %{x}s<extra></extra>"
                ))
            else:
                # Label the midpoint of each target fault interval on the timeline.
                trace.text = [trace.name if task == "Target" and trace.name not in
                              {"Nominal", "Unavailable"} else "" for task in trace.y]
                trace.textposition = "top center"
                trace.textfont = dict(size=11, color="#191835")
        figure.update_layout(
            xaxis_title="simulationTime [s]",
            yaxis=dict(title=None, type="category", categoryorder="array", categoryarray=["Target", "Predicted"]),
            template="plotly_white",
            height=300,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=85, r=25, t=30, b=75),
        )

        violation_counts = {
            column: int(flag.sum()) for column, flag in out_of_range.items() if flag.sum() > 0
        }
        ranked = sorted(violation_counts.items(), key=lambda item: (-item[1], item[0]))

        _, target, known = _target_labels(measurements.index, faults)
        prediction = predicted.to_numpy(dtype=bool)
        counts = {
            "TP": int((known & prediction & target).sum()),
            "TN": int((known & ~prediction & ~target).sum()),
            "FP": int((known & prediction & ~target).sum()),
            "FN": int((known & ~prediction & target).sum()),
        }
        metric_labels = {"TP": "True positives", "TN": "True negatives",
                         "FP": "False positives", "FN": "False negatives"}
        metrics_table = html.Div(
            html.Table([
                html.Caption("Detection counts", className="benchmark_caption"),
                html.Tbody([
                    html.Tr([
                        html.Td([
                            html.Span(name, className="confusion_label"),
                            html.Strong(f"{counts[name]:,}", className="confusion_value"),
                        ], title=metric_labels[name]) for name in row
                    ]) for row in [("TP", "FP"), ("FN", "TN")]
                ]),
            ], className="benchmark_table confusion_table"),
            className="benchmark_card", tabIndex=0, role="region",
            **{"aria-label": "Range monitoring confusion counts"},
        )
        required_columns = set(monitored_columns) | {
            thresholds[column]["state_column"] for column in monitored_columns
            if "state_column" in thresholds[column]
        }
        missing_warning = missing_value_warning(measurements, sorted(required_columns))
        summary = html.Div(
            [
                metrics_table,
                html.P(f"{int((~known).sum()):,} samples excluded because target labels are unavailable.",
                       className="benchmark_legend") if not known.all() else None,
                html.P(f"{len(ranked)} signals exceed their training limits. Select a signal below.")
                if ranked else None,
                html.P(missing_warning,
                       role="alert", style={"color": "#b45309"})
                if missing_warning else None,
            ]
        )
        options = [{"label": f"{column} ({count:,} out-of-range samples)", "value": column}
                   for column, count in ranked]
        return figure, summary, options

    def layout(self, role, analysis, start, end):
        model = self._ensure_model()
        message = self._model_error if model is None else None
        return html.Div([
            html.Div(message, id=self._summary_id),
            dcc.Store(id=self._overview_store_id),
            html.H4("Out-of-range signals"),
            html.Div([
                html.Label("Signal", htmlFor=self._signal_id),
                dcc.Dropdown(id=self._signal_id, options=[], clearable=False,
                             placeholder="No out-of-range signals", className="control_very_wide"),
            ], className="content_control_wide control_very_wide_expandable"),
            html.Div([
                html.Label("Any measurement signal", htmlFor=self._all_signal_id),
                dcc.Dropdown(id=self._all_signal_id, options=[], clearable=True,
                             placeholder="Select an additional signal to inspect",
                             className="control_very_wide"),
            ], className="content_control_wide control_very_wide_expandable"),
            html.P("Complete selected signal. Red markers show samples outside the training limits.",
                   style={"fontSize": "13px", "marginTop": "12px"}),
            dcc.Loading(dcc.Graph(id=self._graph_id, figure=_empty_figure(message or "Select a scenario."))),
        ], style={"padding": "1rem"})

    def _signal_data(self, file_value, signal):
        split_name, scenario_name = split_file(file_value)
        path = scenario_path(self._datasets, split_name, scenario_name)
        model = self._ensure_model()
        if path is None or not signal:
            return None
        bounds = (model or {}).get(signal) if is_ad_signal(signal) else None
        # A state-conditioned signal needs its owning state as well as its values.
        columns = [signal]
        if bounds is not None and bounds.get("state_column") not in (None, signal):
            columns.append(bounds["state_column"])
        frame = pd.read_parquet(path / TABLES["measurements"], columns=columns).sort_index()
        flags = (predict_column_anomalies(frame, {signal: bounds}, warn=False)[signal] if bounds is not None
                 else pd.Series(False, index=frame.index))
        return frame, flags, bounds

    def _detail_figure(self, file_value, signal):
        data = self._signal_data(file_value, signal)
        if data is None:
            return _empty_figure("Select a signal.")
        frame, flags, bounds = data
        series = frame[signal]
        categorical = (bounds.get("type", "numeric") == "categorical" if bounds is not None
                       else not pd.api.types.is_numeric_dtype(series.dtype))
        plotted = series.astype("string").fillna("Missing") if categorical else series.astype(float)
        violations = plotted[flags]
        figure = go.Figure([
            go.Scatter(x=series.index.tolist(), y=plotted.tolist(), name=signal,
                       mode="lines+markers", line=dict(color="#2563eb", width=1.5), marker=dict(size=3),
                       connectgaps=False, hovertemplate="Time: %{x}s<br>Value: %{y}<extra>%{fullData.name}</extra>"),
            go.Scatter(x=violations.index.tolist(), y=violations.tolist(),
                       name="Out of range", mode="markers", marker=dict(color="#dc2626", size=7)),
        ])
        if categorical:
            figure.update_yaxes(type="category")
            if bounds is not None:
                values = ("; ".join(f"{state}: {', '.join(profile['categories'])}"
                                    for state, profile in bounds["states"].items())
                          if "state_column" in bounds else ", ".join(bounds["categories"]))
                figure.add_annotation(text="Training values: " + values,
                                      x=0, y=1.08, xref="paper", yref="paper", showarrow=False)
        elif bounds is not None and "state_column" in bounds:
            state_keys = frame[bounds["state_column"]].astype("string")
            for key, label in [("lower", "Training minimum"), ("upper", "Training maximum")]:
                limits = state_keys.map({state: profile[key] for state, profile in bounds["states"].items()})
                figure.add_trace(go.Scatter(
                    x=series.index.tolist(), y=limits.tolist(), name=label,
                    mode="lines", line=dict(color="#64748b", dash="dash", shape="hv"),
                    connectgaps=False,
                ))
            figure.add_annotation(text=f"Training limits follow {bounds['state_column']}",
                                  x=0, y=1.08, xref="paper", yref="paper", showarrow=False)
        elif bounds is not None:
            for key, label in [("min", "Training minimum"), ("max", "Training maximum")]:
                figure.add_hline(y=bounds[key], line_dash="dash", line_color="#64748b",
                                 annotation_text=f"{label}: {bounds[key]:g}", annotation_position="top left")
            figure.add_hrect(y0=bounds["min"], y1=bounds["max"], fillcolor="#14b8a6", opacity=0.07,
                            line_width=0, layer="below")
        if bounds is None:
            figure.add_annotation(text="No training limits available for this signal",
                                  x=0, y=1.08, xref="paper", yref="paper", showarrow=False)
        if series.isna().any():
            figure.add_annotation(text=f"Warning: {int(series.isna().sum()):,} missing samples; ignored for anomaly scoring",
                                  x=0, y=1.16, xref="paper", yref="paper", showarrow=False,
                                  font={"color": "#b45309"})
        if bounds is not None and "state_column" in bounds and frame[bounds["state_column"]].isna().any():
            count = int(frame[bounds["state_column"]].isna().sum())
            figure.add_annotation(text=f"Warning: {count:,} missing component states; ignored for anomaly scoring",
                                  x=0, y=1.24, xref="paper", yref="paper", showarrow=False,
                                  font={"color": "#b45309"})
        figure.update_layout(template="plotly_white", height=480,
                             xaxis_title="simulationTime [s]", hovermode="closest",
                             legend=dict(orientation="h", y=-0.2), margin=dict(l=65, r=30, t=30, b=90))
        return figure

    @staticmethod
    def _combined_figure(overview, detail, additional_detail=None):
        sources = [go.Figure(overview), detail]
        if additional_detail is not None:
            sources.append(additional_detail)
        figure = make_subplots(rows=len(sources), cols=1, shared_xaxes=True,
                               row_heights=[0.3, 0.7] if len(sources) == 2 else [0.2, 0.4, 0.4],
                               vertical_spacing=0.09)
        for trace in go.Figure(overview).data:
            figure.add_trace(trace, row=1, col=1)
        for row, source in enumerate(sources[1:], start=2):
            for trace in source.data:
                figure.add_trace(trace, row=row, col=1)
            for shape in source.layout.shapes:
                properties = shape.to_plotly_json()
                properties.update(xref=f"x{row} domain", yref=f"y{row}")
                figure.add_shape(properties)
            figure.update_yaxes(type=source.layout.yaxis.type, row=row, col=1)
        for row, source in enumerate(sources, start=1):
            for annotation in source.layout.annotations:
                properties = annotation.to_plotly_json()
                axis = "" if row == 1 else "2"
                properties["xref"] = f"x{axis} domain"
                properties["yref"] = f"y{axis} domain" if annotation.yref == "paper" else f"y{axis}"
                figure.add_annotation(properties)
        figure.update_yaxes(type="category", categoryorder="array",
                            categoryarray=["Target", "Predicted"], row=1, col=1)
        figure.update_xaxes(title_text="simulationTime [s]", row=len(sources), col=1)
        figure.update_layout(template="plotly_white", height=760 if len(sources) == 2 else 1100,
                             hovermode="closest",
                             legend=dict(orientation="h", y=-0.12),
                             margin=dict(l=85, r=30, t=30, b=90))
        return figure

    def register_callbacks(self, dash_app, analysis):
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

        @dash_app.callback(Output(self._overview_store_id, "data"), Output(self._summary_id, "children"),
                           Output(self._signal_id, "options"), Output(self._signal_id, "value"),
                           Output(self._all_signal_id, "options"),
                           Input(self._file_selector_id, "value"))
        def update_scenario(file_value):
            model = self._ensure_model()
            if model is None:
                return {"file": file_value, "figure": _empty_figure(self._model_error or "Model unavailable.").to_dict()}, self._model_error, [], None, []
            split_name, scenario_name = split_file(file_value)
            try:
                frame = load_table(self._datasets, split_name, scenario_name, "measurements")
                faults = load_table(self._datasets, split_name, scenario_name, "faults")
                figure, summary, options = self._overview(frame, model, file_value or "Select a scenario", faults)
                all_options = [{"label": str(column), "value": str(column)} for column in frame] if frame is not None else []
                return {"file": file_value, "figure": figure.to_dict()}, summary, options, options[0]["value"] if options else None, all_options
            except Exception as error:
                return {"file": file_value, "figure": _empty_figure("Could not analyze scenario.").to_dict()}, str(error), [], None, []

        @dash_app.callback(Output(self._graph_id, "figure"), Input(self._overview_store_id, "data"),
                           Input(self._signal_id, "value"), Input(self._all_signal_id, "value"))
        def update_detail(overview, signal, additional_signal=None):
            if not overview:
                return _empty_figure("Select a scenario.")
            try:
                detail = self._detail_figure(overview["file"], signal)
            except (OSError, ValueError, KeyError) as error:
                detail = _empty_figure(f"Could not load signal: {error}")
            additional_detail = None
            if additional_signal:
                try:
                    additional_detail = self._detail_figure(overview["file"], additional_signal)
                except (OSError, ValueError, KeyError) as error:
                    additional_detail = _empty_figure(f"Could not load signal: {error}")
            return self._combined_figure(overview["figure"], detail, additional_detail)

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

    def _model_path(self) -> Path:
        return MODELS_PATH / self.MODEL_FILENAME

    def _ensure_model(self) -> Any:
        """Load saved thresholds and retry on the next visit if unavailable."""
        if self._model is not None:
            return self._model
        model_path = self._model_path()
        try:
            self._model = self._load_model(model_path)
            self._model_error = None
        except FileNotFoundError:
            self._model_error = (
                f"No range-monitoring model found at {model_path}. "
                f"Run python methods/ad/{self.TRAINING_SCRIPT} to train and save it."
            )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._model_error = f"Could not load range-monitoring model: {error}"
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


class RangeMonitoringAll(RangeMonitoring):
    """Range monitoring trained on all training recordings."""


class RangeMonitoring01Pct(RangeMonitoring):
    """Range monitoring trained on 0% and 1% fault recordings."""

    MODEL_FILENAME = "range_monitoring_0_1pct.json"
    TRAINING_SCRIPT = "range_monitoring_0_1pct.py"


class RangeMonitoring0Pct(RangeMonitoring):
    """Range monitoring trained only on 0% fault recordings."""

    MODEL_FILENAME = "range_monitoring_0pct.json"
    TRAINING_SCRIPT = "range_monitoring_0pct.py"
