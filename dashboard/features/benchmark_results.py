"""Benchmark results feature.

Renders the AD and DX benchmark tables (benchmark_ad.json and benchmark_dx.json, produced
by scripts/evaluate_methods.py) as static HTML tables, mirroring the LaTeX
tables used in the paper.
"""

from __future__ import annotations

import json
from typing import Any

from dash import html
from selfx.backend import features

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

AD_RESULTS_PATH = REPO_ROOT / "benchmark_ad.json"
DX_RESULTS_PATH = REPO_ROOT / "benchmark_dx.json"

_NOT_AVAILABLE = "Not evaluated yet. Run scripts/evaluate_methods.py."


def _metric_cell(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _anomaly_detection_table(results: dict[str, dict[str, float]] | None) -> html.Table:
    methods = ["Range Monitoring", "Vanilla Autoencoder"]
    methods.extend(method for method in (results or {}) if method not in methods)
    rows = []
    for method in methods:
        metrics = (results or {}).get(method)
        if metrics is None:
            rows.append(
                html.Tr([html.Td(method), html.Td(_NOT_AVAILABLE, colSpan=3)])
            )
        else:
            rows.append(
                html.Tr(
                    [
                        html.Td(method),
                        html.Td(_metric_cell(metrics.get("BA"))),
                        html.Td(_metric_cell(metrics.get("F1"))),
                        html.Td(_metric_cell(metrics.get("F1_comp"))),
                    ]
                )
            )
    return html.Table(
        [
            html.Caption("Anomaly detection", className="benchmark_caption"),
            html.Thead(
                html.Tr(
                    [html.Th("Method"), html.Th("BA"), html.Th("F1"), html.Th("F1 Comp.")]
                )
            ),
            html.Tbody(rows),
        ],
        className="benchmark_table",
    )


def _diagnosis_table(results: dict[str, dict[str, float]] | None) -> html.Table:
    methods = ["Structural-Knowledge-Based", "Case-Based"]
    methods.extend(method for method in (results or {}) if method not in methods)
    rows = []
    for method in methods:
        metrics = (results or {}).get(method)
        if metrics is None:
            rows.append(
                html.Tr([html.Td(method), html.Td(_NOT_AVAILABLE, colSpan=4)])
            )
        else:
            rows.append(
                html.Tr(
                    [
                        html.Th(method, scope="row"),
                        html.Td(_metric_cell(metrics.get("Loc_BA"))),
                        html.Td(_metric_cell(metrics.get("Fault_BA"))),
                        html.Td(_metric_cell(metrics.get("Loc_F1"))),
                        html.Td(_metric_cell(metrics.get("Fault_F1"))),
                    ]
                )
            )
    return html.Table(
        [
            html.Caption("Fault diagnosis", className="benchmark_caption"),
            html.Thead(
                html.Tr(
                    [
                        html.Th("Method"),
                        html.Th("Loc. BA"),
                        html.Th("Fault BA"),
                        html.Th("Loc. F1"),
                        html.Th("Fault F1"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        className="benchmark_table",
    )


class BenchmarkResults(features.Feature):
    """Static AD/DX benchmark tables computed by scripts/evaluate_methods.py."""

    abstract = False

    def icon(self) -> str:
        return "leaderboard"

    @staticmethod
    def _load_results() -> dict[str, Any] | None:
        results = {}
        for task, path in (("anomaly_detection", AD_RESULTS_PATH), ("diagnosis", DX_RESULTS_PATH)):
            if path.is_file():
                results[task] = json.loads(path.read_text(encoding="utf-8"))
        return results or None

    def layout(self, role: Any, analysis: Any, start: Any, end: Any) -> html.Div:
        results = self._load_results()
        if results is None:
            return html.Div(
                html.P(
                    f"No benchmark results found. Run scripts/evaluate_methods.py to "
                    f"generate {AD_RESULTS_PATH} and {DX_RESULTS_PATH}."
                ),
                style={"padding": "1rem"},
            )

        return html.Div(
            [
                html.Div(
                    [
                        html.H2("Benchmark results"),
                        html.P("Method performance on the held-out test scenarios."),
                    ],
                    className="benchmark_heading",
                ),
                html.Div(
                    _anomaly_detection_table(results.get("anomaly_detection")),
                    className="benchmark_card",
                    tabIndex=0,
                    role="region",
                    **{"aria-label": "Anomaly detection benchmark"},
                ),
                html.Div(
                    _diagnosis_table(results.get("diagnosis")),
                    className="benchmark_card",
                    tabIndex=0,
                    role="region",
                    **{"aria-label": "Fault diagnosis benchmark"},
                ),
                html.P(
                    "BA: balanced accuracy · F1 Comp.: composite F1 · "
                    "Loc.: component localization. Scores range from 0 to 1; higher is better.",
                    className="benchmark_legend",
                ),
            ],
            className="benchmark_results",
        )
