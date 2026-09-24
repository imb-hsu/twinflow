"""Structural-knowledge-based fault diagnosis feature.

Loads the structural knowledge model (models/structural_knowledge.joblib,
trained by scripts/train_structural_knowledge.py) together with the range-
monitoring thresholds (models/range_monitoring.json) that flag anomalous
variables, then maps anomalous variables to owning components and their
possible fault types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from dash import html

from . import dataset_utils as ds
from .model_feature_base import ModelFeatureBase


class StructKnowledgeDX(ModelFeatureBase):
    """Maps anomalous variables to components and their possible faults."""

    abstract = False
    MODEL_FILENAME = "structural_knowledge.joblib"

    def icon(self) -> str:
        return "account_tree"

    def _load_model(self, model_path: Path) -> dict:
        model = joblib.load(model_path)
        thresholds_path = ds.MODELS_PATH / "range_monitoring.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(
                f"{thresholds_path} not found; run scripts/train_range_monitoring.py "
                "first so this feature knows which variables are anomalous."
            )
        model["thresholds"] = json.loads(thresholds_path.read_text(encoding="utf-8"))
        return model

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        import plotly.graph_objects as go

        measurements = ds.load_table(self._datasets, split_name, scenario_name, "measurements")
        if measurements is None:
            return go.Figure(), "No measurements found for this scenario."

        thresholds = model["thresholds"]
        component_type_by_id = model["component_type_by_id"]
        area_by_id = model["area_by_id"]
        faults_by_type = model["faults_by_type"]

        monitored_columns = [column for column in measurements.columns if column in thresholds]
        component_hit_counts: dict[str, int] = {}
        for column in monitored_columns:
            bounds = thresholds[column]
            series = measurements[column]
            out_of_range = (series < bounds["min"]) | (series > bounds["max"])
            hits = int(out_of_range.sum())
            if hits == 0:
                continue
            component_id, _, _ = column.rpartition(".")
            component_hit_counts[component_id or column] = (
                component_hit_counts.get(component_id or column, 0) + hits
            )

        ranked_components = sorted(component_hit_counts.items(), key=lambda item: -item[1])[:15]

        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                x=[component_id for component_id, _ in ranked_components],
                y=[count for _, count in ranked_components],
                name="Anomalous samples",
            )
        )
        figure.update_layout(
            title=f"Implicated components — {split_name}/{scenario_name}",
            xaxis_title="Component",
            yaxis_title="Anomalous samples (summed over its out-of-range variables)",
        )

        rows = []
        for component_id, count in ranked_components:
            component_type = component_type_by_id.get(component_id, "unknown")
            area = area_by_id.get(component_id, "unknown")
            possible_faults = faults_by_type.get(component_type, [])
            rows.append(
                html.Li(
                    f"{component_id} (area {area}, type {component_type}): "
                    f"{count} anomalous samples — possible faults: "
                    f"{', '.join(possible_faults) if possible_faults else 'none known'}"
                )
            )

        summary = html.Div(
            [
                html.P(
                    f"{len(component_hit_counts)} components had at least one "
                    f"out-of-range variable."
                ),
                html.Ul(rows),
            ]
        )
        return figure, summary
