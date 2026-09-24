import json
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, html

from dashboard.features.range_monitoring import RangeMonitoring, anomaly_stateflow


class RangeMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.frame = pd.DataFrame({"speed": [1., 1., 5., 1., 1., 1., 1., 1., -1., 1., 1.],
                                   "normal": [1.] * 11}, index=np.arange(11, dtype=float))
        self.frame.to_parquet(self.path / "measurements.parquet")
        self.feature = RangeMonitoring()
        self.feature._datasets = {"test": {"example": self.path}}
        self.feature._model = {name: {"min": 0., "max": 2.} for name in self.frame}

    def test_missing_model_message_and_recovery(self):
        model_path = self.path / 'range_monitoring.json'
        self.feature._model = None
        with patch.object(self.feature, '_model_path', return_value=model_path):
            layout = self.feature.layout(None, None, None, None)
            self.assertIn('No range-monitoring model found', layout.children[0].children)
            model_path.write_text(json.dumps({'speed': {'type': 'numeric', 'lower': 0., 'upper': 2.}}))
            model = self.feature._ensure_model()
        self.assertEqual(model['speed']['min'], 0.)
        self.assertEqual(model['speed']['max'], 2.)
        self.assertIsNone(self.feature._model_error)

    def test_saved_numeric_and_categorical_model(self):
        model_path = self.path / 'range_monitoring.json'
        model_path.write_text(json.dumps({
            'speed': {'type': 'numeric', 'lower': 0., 'upper': 2.},
            'state': {'type': 'categorical', 'categories': ['idle', 'running']},
            'enabled': {'type': 'categorical', 'categories': ['False']},
        }))
        self.feature._model = self.feature._load_model(model_path)
        frame = pd.DataFrame({'speed': [1., 3., 1.], 'state': ['idle', 'running', 'broken'],
                              'enabled': [False, True, False]}, index=[0., 1., 2.])
        frame.to_parquet(self.path / 'measurements.parquet')
        _, _, options = self.feature._overview(frame, self.feature._model, 'test/example')
        self.assertEqual({option['value'] for option in options}, {'speed', 'state', 'enabled'})
        numeric = self.feature._detail_figure('test/example', 'speed')
        self.assertEqual(list(numeric.data[1].y), [3.])
        category = self.feature._detail_figure('test/example', 'state')
        self.assertEqual(list(category.data[1].y), ['broken'])
        self.assertEqual(category.layout.yaxis.type, 'category')
        category.to_json()

    def test_only_violating_signals_are_offered(self):
        _, _, options = self.feature._overview(self.frame, self.feature._model, "example")
        self.assertEqual([option["value"] for option in options], ["speed"])

    def test_complete_signal_and_all_anomalies(self):
        figure = self.feature._detail_figure("test/example", "speed")
        self.assertEqual(list(figure.data[0].x), list(self.frame.index))
        self.assertEqual(list(figure.data[0].y), self.frame.speed.tolist())
        self.assertEqual(list(figure.data[1].x), [2., 8.])
        self.assertEqual(list(figure.data[1].y), [5., -1.])
        self.assertFalse(figure.data[0].connectgaps)
        self.assertEqual([shape.y0 for shape in figure.layout.shapes[:2]], [0., 2.])

    def test_no_anomalies_and_callback_registration(self):
        figure = self.feature._detail_figure("test/example", "normal")
        self.assertEqual(list(figure.data[0].x), list(self.frame.index))
        self.assertEqual(len(figure.data[1].x), 0)
        self.assertNotIn("context_seconds", self.feature.config)
        app = Dash(__name__)
        app.layout = html.Div([self.feature.topbar_controls(), self.feature.layout(None, None, None, None)])
        self.feature.register_callbacks(app, None)
        self.feature.register_callbacks(app, None)
        self.assertEqual(len(app.callback_map), 2)
        self.assertEqual(app.server.test_client().get("/_dash-layout").status_code, 200)
        callbacks = {entry["callback"].__wrapped__.__name__: entry["callback"].__wrapped__
                     for entry in app.callback_map.values()}
        overview, _, _, _ = callbacks["update_scenario"]("test/example")
        figure = callbacks["update_detail"](overview, "speed")
        signal_trace = next(trace for trace in figure.data if trace.name == "speed")
        self.assertEqual(len(signal_trace.x), len(self.frame))
        self.assertEqual(signal_trace.yaxis, "y2")
        self.assertEqual(figure.layout.xaxis.matches, "x2")
        self.assertTrue(all(shape.yref == "y2" for shape in figure.layout.shapes))
        self.assertTrue(any(trace.yaxis == "y" for trace in figure.data))
        switched = callbacks["update_detail"](overview, "normal")
        self.assertEqual(next(trace for trace in switched.data if trace.yaxis == "y2").name, "normal")

    def test_all_64_anomalies_remain_marked(self):
        frame = pd.DataFrame({"speed": np.ones(640)}, index=np.arange(640, dtype=float))
        spikes = np.arange(5, 640, 10)
        frame.loc[spikes, "speed"] = 5.
        frame.to_parquet(self.path / "measurements.parquet")
        figure = self.feature._detail_figure("test/example", "speed")
        self.assertEqual(len(figure.data[0].x), 640)
        self.assertEqual(list(figure.data[1].x), spikes.tolist())

    def test_target_alignment_and_single_sample_anomalies(self):
        faults = pd.DataFrame({"fault": pd.array([True, False, True], dtype="boolean"),
                               "severity": [0.5, 0.5, 0.5]}, index=[3., 0., 1.])
        timeline = anomaly_stateflow([0., 1., 2., 3.], [False, True, False, True], faults)
        predictions = timeline[(timeline.Task == "Predicted") & (timeline.State == "Anomaly")]
        self.assertEqual(list(zip(predictions.Start, predictions.Finish)), [(1., 2.), (3., 4.)])
        target = timeline[timeline.Task == "Target"]
        self.assertEqual(target.State.tolist(), ["Nominal", "fault", "Unavailable", "fault"])
        missing = anomaly_stateflow([0., 1.], [False, False], None)
        self.assertEqual(missing[missing.Task == "Target"].State.tolist(), ["Unavailable"])

    def test_overview_stateflow_contains_predictions_and_targets(self):
        faults = pd.DataFrame({"fault": [False] * 11}, index=self.frame.index)
        faults.loc[3., "fault"] = True
        figure, _, _ = self.feature._overview(self.frame, self.feature._model, "example", faults)
        lanes = {y for trace in figure.data for y in trace.y if y is not None}
        self.assertEqual(lanes, {"Predicted", "Target"})
        self.assertEqual({trace.name for trace in figure.data}, {"Nominal", "Anomaly", "fault"})
        figure.to_json()

    def test_exact_simultaneous_fault_labels_and_confusion_counts(self):
        faults = pd.DataFrame({"AMR_1.MotorFault": pd.array([False] * 11, dtype="boolean"),
                               "AMR_2.EmergencyStop": pd.array([False] * 11, dtype="boolean")},
                              index=self.frame.index)
        faults.loc[[2., 3.], "AMR_1.MotorFault"] = True
        faults.loc[2., "AMR_2.EmergencyStop"] = True
        faults.loc[0., :] = pd.NA
        figure, summary, _ = self.feature._overview(self.frame, self.feature._model, "example", faults)
        table = summary.children[0].children
        self.assertIn("benchmark_table", table.className)
        rows = table.children[1].children
        self.assertEqual([[cell.children[0].children for cell in row.children] for row in rows],
                         [["TP", "FP"], ["FN", "TN"]])
        self.assertEqual([[cell.children[1].children for cell in row.children] for row in rows],
                         [["1", "1"], ["1", "7"]])
        self.assertEqual(summary.children[1].children,
                         "1 samples excluded because target labels are unavailable.")
        names = {trace.name for trace in figure.data}
        self.assertIn("AMR_1.MotorFault + AMR_2.EmergencyStop", names)
        self.assertIn("AMR_1.MotorFault", names)
        labels = [label for trace in figure.data if trace.text is not None for label in trace.text]
        self.assertIn("AMR_1.MotorFault + AMR_2.EmergencyStop", labels)


if __name__ == "__main__":
    unittest.main()
