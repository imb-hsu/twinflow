"""Case-based fault diagnosis feature.

Loads the reference case bank (models/case_based.joblib, trained by
scripts/train_case_based.py) and, for the selected scenario, finds the
nearest reference case for a sample of timestamps to diagnose the most
likely fault label at each point in time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from dash import html
from sklearn.neighbors import NearestNeighbors

from . import dataset_utils as ds
from .model_feature_base import ModelFeatureBase

# Diagnosing every sample is unnecessary; a coarser stride keeps the plot readable.
ROW_STRIDE = 20


class CaseBasedDX(ModelFeatureBase):
    """Nearest-neighbor diagnosis against a bank of labeled reference cases."""

    abstract = False
    MODEL_FILENAME = "case_based.joblib"

    def icon(self) -> str:
        return "compare_arrows"

    def _load_model(self, model_path: Path) -> dict:
        artifact = joblib.load(model_path)
        vectors = np.stack([case["vector"] for case in artifact["cases"]])
        neighbors = NearestNeighbors(n_neighbors=1).fit(vectors)
        artifact["_neighbors"] = neighbors
        return artifact

    def _analyze(self, split_name: str, scenario_name: str, model: dict):
        import plotly.graph_objects as go

        measurements = ds.load_table(self._datasets, split_name, scenario_name, "measurements")
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

        sampled = measurements.iloc[::ROW_STRIDE]
        X = sampled[columns].fillna(sampled[columns].mean()).to_numpy()
        X_scaled = model["scaler"].transform(X)

        distances, indices = model["_neighbors"].kneighbors(X_scaled)
        cases = model["cases"]
        fault_labels = [cases[index]["fault_label"] for index in indices[:, 0]]
        distances = distances[:, 0]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=sampled.index,
                y=fault_labels,
                mode="markers",
                marker=dict(size=6),
                name="Diagnosed fault",
                customdata=distances,
                hovertemplate="t=%{x}<br>%{y}<br>distance=%{customdata:.2f}<extra></extra>",
            )
        )
        figure.update_layout(
            title=f"Case-based diagnosis — {split_name}/{scenario_name}",
            xaxis_title="simulationTime [s]",
            yaxis_title="Nearest reference case",
        )

        label_counts: dict[str, int] = {}
        for label in fault_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        top_labels = sorted(label_counts.items(), key=lambda item: -item[1])

        summary = html.Div(
            [
                html.P(f"Diagnosed {len(sampled)} sampled timestamps against {len(cases)} reference cases."),
                html.Ul([html.Li(f"{label}: {count} samples") for label, count in top_labels]),
            ]
        )
        return figure, summary
