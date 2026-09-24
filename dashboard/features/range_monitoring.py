"""Range-monitoring anomaly detection feature.

Loads per-column min/max thresholds (models/range_monitoring.json, trained by
scripts/train_range_monitoring.py) and flags out-of-range samples in the
selected scenario's measurements.

The overview compares predicted and target anomaly states; the signal inspector shows the raw
complete signal and training bounds, with anomalies marked in red.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Input, Output, dcc, html

try:
    from ml4cps import plot_stateflow
except ImportError:
    from ml4cps.vis import plot_stateflow

from . import dataset_utils as ds
from .model_feature_base import ModelFeatureBase


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
        columns = [column for column in faults if pd.api.types.is_bool_dtype(faults[column].dtype)]
        if columns:
            aligned = faults[columns].reindex(index)
            active = aligned.fillna(False).any(axis=1).to_numpy(dtype=bool)
            known = aligned.notna().all(axis=1).to_numpy(dtype=bool)
            target_states[:] = "Nominal"
            for column in columns:
                mask = aligned[column].fillna(False).to_numpy(dtype=bool)
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


class RangeMonitoring(ModelFeatureBase):
    """Anomaly score based on per-column min/max range violations."""

    abstract = False
    MODEL_FILENAME = "range_monitoring.json"

    def _set_component_ids(self):
        super()._set_component_ids()
        prefix = self._id_prefix()
        self._signal_id = f"{prefix}-signal"
        self._overview_store_id = f"{prefix}-overview"

    def icon(self) -> str:
        return "rule"

    def _load_model(self, model_path: Path) -> dict:
        return json.loads(model_path.read_text(encoding="utf-8"))

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        measurements = ds.load_table(self._datasets, split_name, scenario_name, "measurements")
        faults = ds.load_table(self._datasets, split_name, scenario_name, "faults")
        figure, summary, _ = self._overview(measurements, model, f"{split_name}/{scenario_name}", faults)
        return figure, summary

    def _overview(self, measurements, thresholds, title, faults=None):
        if measurements is None:
            return _empty_figure("No measurements found."), "No measurements found for this scenario.", []

        monitored_columns = [column for column in measurements.columns if column in thresholds]
        if not monitored_columns:
            return _empty_figure("No monitored signals found."), "None of the trained columns are present in this scenario.", []

        measurements = measurements.sort_index()
        out_of_range = {}
        for column in monitored_columns:
            bounds = thresholds[column]
            series = measurements[column]
            out_of_range[column] = ((series < bounds["min"]) | (series > bounds["max"])).fillna(False)

        predicted = pd.DataFrame(out_of_range).any(axis=1)
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
        summary = html.Div(
            [
                metrics_table,
                html.P(f"{int((~known).sum()):,} samples excluded because target labels are unavailable.",
                       className="benchmark_legend") if not known.all() else None,
                html.P(f"{len(ranked)} signals exceed their training limits. Select a signal below.")
                if ranked else None,
            ]
        )
        options = [{"label": f"{column} ({count:,} out-of-range samples)", "value": column}
                   for column, count in ranked]
        return figure, summary, options

    def layout(self, role, analysis, start, end):
        return html.Div([
            html.Div(id=self._summary_id),
            dcc.Store(id=self._overview_store_id),
            html.H4("Out-of-range signals"),
            html.Div([
                html.Label("Signal", htmlFor=self._signal_id),
                dcc.Dropdown(id=self._signal_id, options=[], clearable=False,
                             placeholder="No out-of-range signals", className="control_very_wide"),
            ], className="content_control_wide control_very_wide_expandable"),
            html.P("Complete selected signal. Red markers show samples outside the training limits.",
                   style={"fontSize": "13px", "marginTop": "12px"}),
            dcc.Loading(dcc.Graph(id=self._graph_id, figure=_empty_figure("Select a scenario."))),
        ], style={"padding": "1rem"})

    def _signal_data(self, file_value, signal):
        split_name, scenario_name = ds.split_file(file_value)
        path = ds.scenario_path(self._datasets, split_name, scenario_name)
        model = self._ensure_model()
        if path is None or model is None or signal not in model:
            return None
        # Read the complete selected signal without downsampling.
        frame = pd.read_parquet(path / ds.TABLES["measurements"], columns=[signal]).sort_index()
        bounds = model[signal]
        flags = ((frame[signal] < bounds["min"]) | (frame[signal] > bounds["max"])).fillna(False)
        return frame, flags, bounds

    def _detail_figure(self, file_value, signal):
        data = self._signal_data(file_value, signal)
        if data is None:
            return _empty_figure("Select an out-of-range signal.")
        frame, flags, bounds = data
        series = frame[signal]
        violations = series[flags]
        figure = go.Figure([
            go.Scatter(x=series.index.tolist(), y=series.astype(float).tolist(), name=signal,
                       mode="lines+markers", line=dict(color="#2563eb", width=1.5), marker=dict(size=3),
                       connectgaps=False, hovertemplate="Time: %{x}s<br>Value: %{y}<extra>%{fullData.name}</extra>"),
            go.Scatter(x=violations.index.tolist(), y=violations.astype(float).tolist(),
                       name="Out of range", mode="markers", marker=dict(color="#dc2626", size=7)),
        ])
        for key, label in [("min", "Training minimum"), ("max", "Training maximum")]:
            figure.add_hline(y=bounds[key], line_dash="dash", line_color="#64748b",
                             annotation_text=f"{label}: {bounds[key]:g}", annotation_position="top left")
        figure.add_hrect(y0=bounds["min"], y1=bounds["max"], fillcolor="#14b8a6", opacity=0.07,
                        line_width=0, layer="below")
        figure.update_layout(template="plotly_white", height=480,
                             xaxis_title="simulationTime [s]", hovermode="x unified",
                             legend=dict(orientation="h", y=-0.2), margin=dict(l=65, r=30, t=30, b=90))
        return figure

    @staticmethod
    def _combined_figure(overview, detail):
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.3, 0.7], vertical_spacing=0.09)
        for trace in go.Figure(overview).data:
            figure.add_trace(trace, row=1, col=1)
        for trace in detail.data:
            figure.add_trace(trace, row=2, col=1)
        for shape in detail.layout.shapes:
            properties = shape.to_plotly_json()
            properties.update(xref="x2 domain", yref="y2")
            figure.add_shape(properties)
        for row, source in [(1, go.Figure(overview)), (2, detail)]:
            for annotation in source.layout.annotations:
                properties = annotation.to_plotly_json()
                axis = "" if row == 1 else "2"
                properties["xref"] = f"x{axis} domain"
                properties["yref"] = f"y{axis} domain" if annotation.yref == "paper" else f"y{axis}"
                figure.add_annotation(properties)
        figure.update_yaxes(type="category", categoryorder="array",
                            categoryarray=["Target", "Predicted"], row=1, col=1)
        figure.update_xaxes(title_text="simulationTime [s]", row=2, col=1)
        figure.update_layout(template="plotly_white", height=760, hovermode="x unified",
                             legend=dict(orientation="h", y=-0.12),
                             margin=dict(l=85, r=30, t=30, b=90))
        return figure

    def register_callbacks(self, dash_app, analysis):
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

        @dash_app.callback(Output(self._overview_store_id, "data"), Output(self._summary_id, "children"),
                           Output(self._signal_id, "options"), Output(self._signal_id, "value"),
                           Input(self._file_selector_id, "value"))
        def update_scenario(file_value):
            model = self._ensure_model()
            if model is None:
                return {"file": file_value, "figure": _empty_figure("Model unavailable.").to_dict()}, self._model_error, [], None
            split_name, scenario_name = ds.split_file(file_value)
            try:
                frame = ds.load_table(self._datasets, split_name, scenario_name, "measurements")
                faults = ds.load_table(self._datasets, split_name, scenario_name, "faults")
                figure, summary, options = self._overview(frame, model, file_value or "Select a scenario", faults)
                return {"file": file_value, "figure": figure.to_dict()}, summary, options, options[0]["value"] if options else None
            except Exception as error:
                return {"file": file_value, "figure": _empty_figure("Could not analyze scenario.").to_dict()}, str(error), [], None

        @dash_app.callback(Output(self._graph_id, "figure"), Input(self._overview_store_id, "data"),
                           Input(self._signal_id, "value"))
        def update_detail(overview, signal):
            if not overview:
                return _empty_figure("Select a scenario.")
            try:
                detail = self._detail_figure(overview["file"], signal)
            except (OSError, ValueError, KeyError) as error:
                detail = _empty_figure(f"Could not load signal: {error}")
            return self._combined_figure(overview["figure"], detail)
