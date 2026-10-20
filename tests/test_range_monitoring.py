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

    def test_current_position_is_inspectable_but_not_monitored(self):
        column = "RB_1.Axis4.CurrentPosition"
        frame = self.frame.assign(**{column: 999.})
        frame.to_parquet(self.path / "measurements.parquet")
        self.feature._model[column] = {"min": 0., "max": 1.}
        _, _, options = self.feature._overview(frame, self.feature._model, "example")
        self.assertNotIn(column, {option["value"] for option in options})
        detail = self.feature._detail_figure("test/example", column)
        self.assertEqual(list(detail.data[0].y), [999.] * len(frame))
        self.assertEqual(len(detail.data[1].x), 0)

    def test_missing_signals_warn_without_hiding_observed_violations(self):
        frame = pd.DataFrame({'speed': [None, 3., None],
                              'state': pd.array([None, None, 'Unexpected'], dtype='string'),
                              'enabled': pd.array([None, True, False], dtype='boolean')})
        thresholds = {'speed': {'type': 'numeric', 'min': 0., 'max': 2.},
                      'state': {'type': 'categorical', 'categories': ['Idle']},
                      'enabled': {'type': 'categorical', 'categories': ['False']}}
        for column in frame:
            flags = self.feature._violation_flags(frame[column], thresholds[column])
            self.assertFalse(flags.iloc[0])
        figure, summary, options = self.feature._overview(frame, thresholds, 'missing')
        self.assertEqual({option['value'] for option in options}, set(frame.columns))
        warning = summary.children[-1]
        self.assertEqual(warning.role, 'alert')
        self.assertIn('speed: 2', warning.children)
        self.assertIn('state: 2', warning.children)
        frame.to_parquet(self.path / 'measurements.parquet')
        self.feature._model = thresholds
        detail = self.feature._detail_figure('test/example', 'speed')
        self.assertEqual(list(detail.data[1].y), [3.])
        self.assertTrue(any('2 missing samples' in a.text for a in detail.layout.annotations))

    def test_complete_signal_and_all_anomalies(self):
        figure = self.feature._detail_figure("test/example", "speed")
        self.assertEqual(list(figure.data[0].x), list(self.frame.index))
        self.assertEqual(list(figure.data[0].y), self.frame.speed.tolist())
        self.assertEqual(list(figure.data[1].x), [2., 8.])
        self.assertEqual(list(figure.data[1].y), [5., -1.])
        self.assertFalse(figure.data[0].connectgaps)
        self.assertEqual([shape.y0 for shape in figure.layout.shapes[:2]], [0., 2.])

    def test_state_conditioned_dashboard_matches_backend_and_plots_switching_limits(self):
        from methods.ad.range_monitoring import predict_column_anomalies
        signal = "CC_1.CurrentSpeed"
        state = "CC_1.CurrentState"
        frame = pd.DataFrame({signal: [0., 10., 0., 5.], state: [False, True, True, False]})
        frame.to_parquet(self.path / "measurements.parquet")
        path = self.path / "model.json"
        path.write_text(json.dumps({signal: {
            "type": "numeric", "state_column": state, "states": {
                "False": {"type": "numeric", "lower": 0., "upper": 0.},
                "True": {"type": "numeric", "lower": 9., "upper": 11.},
            },
        }}))
        self.feature._model = self.feature._load_model(path)
        expected = predict_column_anomalies(frame, self.feature._model)[signal]
        self.assertEqual(expected.tolist(), [False, False, True, True])
        detail = self.feature._detail_figure("test/example", signal)
        self.assertEqual(list(detail.data[1].x), frame.index[expected].tolist())
        self.assertEqual(list(detail.data[2].y), [0., 9., 9., 0.])
        self.assertEqual(list(detail.data[3].y), [0., 11., 11., 0.])
        overview, _, options = self.feature._overview(frame, self.feature._model, "example")
        self.assertEqual([option["value"] for option in options], [signal])
        self.feature._combined_figure(overview, detail).to_json()

    def test_state_conditioned_categorical_plot_and_missing_state_warning(self):
        signal, state = "AMR_1.Mode", "AMR_1.CurrentState"
        frame = pd.DataFrame({signal: ["Parked", "Drive", "Parked", "Other", "Drive"],
                              state: ["Idle", "Moving", "Moving", "Service", None]})
        frame.to_parquet(self.path / "measurements.parquet")
        self.feature._model = {signal: {
            "type": "categorical", "state_column": state, "states": {
                "Idle": {"type": "categorical", "categories": ["Parked"]},
                "Moving": {"type": "categorical", "categories": ["Drive"]},
            },
        }}
        detail = self.feature._detail_figure("test/example", signal)
        self.assertEqual(list(detail.data[1].x), [2, 3])
        self.assertTrue(any("1 missing component states" in annotation.text for annotation in detail.layout.annotations))
        self.assertTrue(any("Idle: Parked" in annotation.text for annotation in detail.layout.annotations))
        detail.to_json()

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
        overview, _, _, _, all_options = callbacks["update_scenario"]("test/example")
        self.assertEqual({option["value"] for option in all_options}, {"speed", "normal"})
        figure = callbacks["update_detail"](overview, "speed")
        signal_trace = next(trace for trace in figure.data if trace.name == "speed")
        self.assertEqual(len(signal_trace.x), len(self.frame))
        self.assertEqual(signal_trace.yaxis, "y2")
        self.assertEqual(figure.layout.xaxis.matches, "x2")
        self.assertEqual(figure.layout.hovermode, "closest")
        self.assertTrue(all(shape.yref == "y2" for shape in figure.layout.shapes))
        self.assertTrue(any(trace.yaxis == "y" for trace in figure.data))
        switched = callbacks["update_detail"](overview, "normal")
        self.assertEqual(next(trace for trace in switched.data if trace.yaxis == "y2").name, "normal")
        additional = callbacks["update_detail"](overview, "speed", "normal")
        self.assertEqual(next(trace for trace in additional.data if trace.yaxis == "y3").name, "normal")
        self.assertEqual(additional.layout.xaxis.matches, "x3")
        self.assertEqual(additional.layout.hovermode, "closest")

    def test_any_signal_without_training_limits_can_be_plotted(self):
        frame = self.frame.assign(untrained=3., state="Idle")
        frame.to_parquet(self.path / "measurements.parquet")
        for signal in ("untrained", "state"):
            detail = self.feature._detail_figure("test/example", signal)
            self.assertEqual(list(detail.data[0].y), frame[signal].tolist())
            self.assertEqual(len(detail.data[1].x), 0)
            self.assertEqual(len(detail.layout.shapes), 0)
            self.assertIn("No training limits", detail.layout.annotations[0].text)
            combined = self.feature._combined_figure({}, detail)
            self.assertEqual(combined.layout.yaxis2.type, "category" if signal == "state" else None)
            combined.to_json()

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
                               "severity": [0.0, 0.0, 0.0]}, index=[3., 0., 1.])
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

    def test_numeric_faults_and_damping_baseline_are_in_target(self):
        faults = pd.DataFrame({
            "RB_4.EmergencyStop": [0., 1., 0., 0., np.nan],
            "AMR_1.SpeedCalibrationFault": [0., 0.904, -0.5, 0., 0.],
            "RS_2.IncreasedDampingFault": [0.1, 0.1, 0.171, 0., 0.1],
            "Unrecorded.Fault": [np.nan] * 5,
        }, index=np.arange(5, dtype=float))
        timeline = anomaly_stateflow(faults.index, [False] * 5, faults)
        target = timeline[timeline.Task == "Target"]
        self.assertEqual(target.State.tolist(), [
            "Nominal", "RB_4.EmergencyStop + AMR_1.SpeedCalibrationFault",
            "AMR_1.SpeedCalibrationFault + RS_2.IncreasedDampingFault",
            "Nominal", "Unavailable",
        ])

    def test_switching_scenario_loads_its_own_numeric_target_faults(self):
        other = self.path / "other"
        other.mkdir()
        self.frame.to_parquet(other / "measurements.parquet")
        for path, name in [(self.path, "RB_4.EmergencyStop"),
                           (other, "AMR_1.SpeedCalibrationFault")]:
            faults = pd.DataFrame({name: [0.] * len(self.frame)}, index=self.frame.index)
            faults.loc[3., name] = 0.9
            faults.to_parquet(path / "faults.parquet")
        self.feature._datasets["test"]["other"] = other
        app = Dash(__name__)
        self.feature.register_callbacks(app, None)
        callback = next(entry["callback"].__wrapped__ for entry in app.callback_map.values()
                        if entry["callback"].__wrapped__.__name__ == "update_scenario")
        for scenario, expected, excluded in [
            ("example", "RB_4.EmergencyStop", "AMR_1.SpeedCalibrationFault"),
            ("other", "AMR_1.SpeedCalibrationFault", "RB_4.EmergencyStop"),
        ]:
            overview, summary, _, _, _ = callback(f"test/{scenario}")
            names = {trace["name"] for trace in overview["figure"]["data"]}
            self.assertIn(expected, names)
            self.assertNotIn(excluded, names)
            rows = summary.children[0].children.children[1].children
            self.assertEqual([[cell.children[1].children for cell in row.children] for row in rows],
                             [["0", "2"], ["1", "8"]])

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
