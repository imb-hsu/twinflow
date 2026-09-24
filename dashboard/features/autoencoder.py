"""Autoencoder-based anomaly detection feature.

Loads the trained scikit-learn autoencoder (models/autoencoder.joblib, trained
by scripts/train_autoencoder.py) and plots its per-sample reconstruction error
against the learned threshold for the selected scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from dash import html

from . import dataset_utils as ds
from .model_feature_base import ModelFeatureBase


class AutoencoderAD(ModelFeatureBase):
    """Anomaly score based on autoencoder reconstruction error."""

    abstract = False
    MODEL_FILENAME = "autoencoder.joblib"

    def icon(self) -> str:
        return "psychology"

    def _load_model(self, model_path: Path) -> dict:
        return joblib.load(model_path)

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

        # Reuse the scaler's training-time column means to fill gaps consistently.
        fill_values = dict(zip(columns, model["scaler"].mean_))
        X = measurements[columns].fillna(value=fill_values).to_numpy()
        X_scaled = model["scaler"].transform(X)
        reconstructed = model["model"].predict(X_scaled)
        reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        threshold = model["threshold"]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=measurements.index,
                y=reconstruction_error,
                mode="lines",
                name="Reconstruction error",
            )
        )
        figure.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="threshold")
        figure.update_layout(
            title=f"Autoencoder anomaly score — {split_name}/{scenario_name}",
            xaxis_title="simulationTime [s]",
            yaxis_title="Reconstruction error",
        )

        anomalous = reconstruction_error > threshold
        summary = html.Div(
            html.P(
                f"{int(anomalous.sum())}/{len(measurements)} samples exceed the "
                f"reconstruction-error threshold ({threshold:.4f})."
            )
        )
        return figure, summary
