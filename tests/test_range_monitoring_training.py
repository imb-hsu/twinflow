import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from methods.ad import range_monitoring, range_monitoring_0_1pct, range_monitoring_0pct
from dashboard.features.range_monitoring import RangeMonitoringAll, RangeMonitoring01Pct, RangeMonitoring0Pct
from scripts.evaluate_methods import evaluate_anomaly_detection


class RangeMonitoringTrainingTests(unittest.TestCase):
    def test_saved_range_thresholds_ignore_positions_at_prediction_time(self):
        thresholds = {"speed": {"type": "numeric", "lower": 0., "upper": 2.},
                      "RB_1.Axis4.CurrentPosition": {"type": "numeric", "lower": 0., "upper": 1.}}
        frame = pd.DataFrame({"speed": [1., 3.]})
        flags = range_monitoring.predict_column_anomalies(frame, thresholds)
        self.assertEqual(flags.columns.tolist(), ["speed"])
        self.assertEqual(flags.any(axis=1).tolist(), [False, True])

    def test_legacy_autoencoder_with_positions_requires_retraining(self):
        from methods.ad.autoencoder import encode_measurements
        from dashboard.features.autoencoder import encode_measurements as dashboard_encode
        for encode in (encode_measurements, dashboard_encode):
            with self.assertRaisesRegex(ValueError, "retrain"):
                encode(pd.DataFrame(), {"columns": ["RB_1.Axis4.CurrentPosition"]})

    def test_variants_learn_only_from_their_selected_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "data" / "training"
            for percentage, value in [(0, 1.), (1, 2.), (10, 10.)]:
                path = training / f"{percentage}_0pct_faults_rep_1"
                path.mkdir(parents=True)
                pd.DataFrame({"speed": [0., value], "state": ["idle", str(value)],
                              "enabled": [False, percentage > 0],
                              "CC_1.CurrentState": [False, True],
                              "CC_1.CurrentSpeed": [0., value],
                              "RB_1.Axis4.CurrentPosition": [0., value * 1000],
                              "Machine.X.CurrentPosition": [None, None]}).to_parquet(path / "measurements.parquet")
            (training / "notes").mkdir()
            # A non-scenario folder is excluded from the percentage-filtered variants.
            self.assertEqual(len(range_monitoring.training_paths(training, (0, 1))), 2)
            (training / "notes").rmdir()
            variants = [(range_monitoring, 10.), (range_monitoring_0_1pct, 2.),
                        (range_monitoring_0pct, 1.)]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(range_monitoring, "REPO_ROOT", root))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                for module, _ in variants:
                    stack.enter_context(patch.object(module, "MODEL_PATH", root / "models" / module.MODEL_PATH.name))
                for module, maximum in variants:
                    module.train()
                    model = json.loads(module.MODEL_PATH.read_text())
                    self.assertNotIn("RB_1.Axis4.CurrentPosition", model)
                    self.assertNotIn("Machine.X.CurrentPosition", model)
                    self.assertEqual(model["speed"], {"type": "numeric", "lower": 0., "upper": maximum})
                    self.assertEqual(set(model["state"]["categories"]),
                                     {"idle"} | {str(v) for v in (1., 2., 10.) if v <= maximum})
                    self.assertEqual(model["enabled"]["categories"],
                                     ["False"] if maximum == 1 else ["False", "True"])
                    self.assertEqual(model["CC_1.CurrentSpeed"]["state_column"], "CC_1.CurrentState")
                    self.assertEqual(model["CC_1.CurrentSpeed"]["states"]["False"]["upper"], 0.)
                    self.assertEqual(model["CC_1.CurrentSpeed"]["states"]["True"]["upper"], maximum)
                    frame = pd.DataFrame({"speed": [maximum, maximum + 1],
                                          "state": ["idle"] * 2, "enabled": [False] * 2,
                                          "CC_1.CurrentState": [True, True],
                                          "CC_1.CurrentSpeed": [maximum, maximum + 1]})
                    self.assertEqual(module.predict(frame).tolist(), [False, True])
                with patch("scripts.evaluate_methods.REPO_ROOT", Path(range_monitoring.__file__).resolve().parents[2]):
                    scores = evaluate_anomaly_detection({"sample": {
                        "measurements": pd.DataFrame({"speed": [0., 3.], "state": ["idle"] * 2,
                                                      "enabled": [False] * 2,
                                                      "CC_1.CurrentState": [False, True],
                                                      "CC_1.CurrentSpeed": [0., 3.]}),
                        "faults": pd.DataFrame({"fault": [False, True]}),
                    }}, existing_results={"Vanilla Autoencoder": {}})
                self.assertEqual(set(scores), {module.METHOD_NAME for module, _ in variants})
                self.assertEqual(scores[range_monitoring.METHOD_NAME]["fn"], 1)
                self.assertEqual(scores[range_monitoring_0pct.METHOD_NAME]["tp"], 1)
            with self.assertRaisesRegex(ValueError, "No training recordings"):
                range_monitoring.training_paths(training, (99,))

    def test_dashboard_variants_use_distinct_models_and_component_ids(self):
        variants = [RangeMonitoringAll(), RangeMonitoring01Pct(), RangeMonitoring0Pct()]
        self.assertEqual(len({feature._model_path() for feature in variants}), 3)
        self.assertEqual(len({feature._graph_id for feature in variants}), 3)
        self.assertEqual(len({feature._file_selector_id for feature in variants}), 3)


class StateConditionedTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def train(self, frame):
        folder = self.root / "data/training/0_0pct_faults_rep_1"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / "measurements.parquet")
        model_path = self.root / "model.json"
        with patch.object(range_monitoring, "REPO_ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            range_monitoring.train(model_path=model_path)
        return json.loads(model_path.read_text())

    def test_boolean_states_have_independent_numeric_and_categorical_profiles(self):
        model = self.train(pd.DataFrame({
            "CC_1.CurrentState": [False, False, True, True],
            "CC_1.CurrentSpeed": [0., 0., 10., 12.],
            "CC_1.CurrentDirection": ["Stopped", "Stopped", "Forward", "Forward"],
            "CC_2.CurrentState": [True, True, False, False],
            "CC_2.CurrentSpeed": [20., 22., 0., 0.],
            "unscoped": [0., 1., 2., 3.],
        }))
        config = model["CC_1.CurrentSpeed"]
        self.assertEqual(config["states"]["False"], {"type": "numeric", "lower": 0., "upper": 0.})
        self.assertEqual(config["states"]["True"], {"type": "numeric", "lower": 10., "upper": 12.})
        self.assertEqual(model["unscoped"], {"type": "numeric", "lower": 0., "upper": 3.})
        probe = pd.DataFrame({
            "CC_1.CurrentState": [False, True, True, False],
            "CC_1.CurrentSpeed": [0., 11., 0., 11.],
            "CC_1.CurrentDirection": ["Stopped", "Forward", "Stopped", "Forward"],
        })
        flags = range_monitoring.predict_column_anomalies(probe, {
            key: model[key] for key in probe
        })
        self.assertEqual(flags["CC_1.CurrentSpeed"].tolist(), [False, False, True, True])
        self.assertEqual(flags["CC_1.CurrentDirection"].tolist(), [False, False, True, True])

    def test_nested_components_axes_vectors_and_named_amr_states(self):
        model = self.train(pd.DataFrame({
            "CLT_15.CurrentState": [False, True, False],
            "CLT_15.Lift.CurrentState": [True, False, True],
            "CLT_15.Lift.CurrentSpeed": [5., 0., 6.],
            "AMR_1.CurrentState": ["Idle", "Moving", "Turning"],
            "AMR_1.CurrentDirection.X": [0., 1., -1.],
            "AMR_1.AMP.CurrentState": [True, False, True],
            "AMR_1.AMP.CurrentSpeed": [2., 0., 3.],
            "PR_7.X.CurrentState": [False, True, True],
            "PR_7.X.CurrentForce.X": [0., 10., 12.],
        }))
        for signal, state in [
            ("CLT_15.Lift.CurrentSpeed", "CLT_15.Lift.CurrentState"),
            ("AMR_1.CurrentDirection.X", "AMR_1.CurrentState"),
            ("AMR_1.AMP.CurrentSpeed", "AMR_1.AMP.CurrentState"),
            ("PR_7.X.CurrentForce.X", "PR_7.X.CurrentState"),
        ]:
            self.assertEqual(model[signal]["state_column"], state)
        self.assertEqual(model["CLT_15.Lift.CurrentSpeed"]["states"]["True"]["lower"], 5.)
        self.assertEqual(set(model["AMR_1.CurrentDirection.X"]["states"]), {"Idle", "Moving", "Turning"})

    def test_unseen_states_are_anomalous_and_missing_states_are_ignored(self):
        model = self.train(pd.DataFrame({
            "AMR_1.CurrentState": ["Idle", "Moving"],
            "AMR_1.CurrentSpeed": [0., 10.],
        }))
        probe = pd.DataFrame({
            "AMR_1.CurrentState": pd.array(["Idle", "Moving", "Service", None, "Moving"], dtype="string"),
            "AMR_1.CurrentSpeed": [0., 10., 10., 999., None],
        })
        with self.assertWarnsRegex(RuntimeWarning, "AMR_1.CurrentState: 1"):
            flags = range_monitoring.predict_column_anomalies(probe, model)
        self.assertEqual(flags["AMR_1.CurrentSpeed"].tolist(), [False, False, True, False, False])
        self.assertEqual(flags["AMR_1.CurrentState"].tolist(), [False, False, True, False, False])
        with self.assertRaisesRegex(ValueError, "AMR_1.CurrentState"):
            range_monitoring.predict_column_anomalies(probe.drop(columns="AMR_1.CurrentState"), {
                "AMR_1.CurrentSpeed": model["AMR_1.CurrentSpeed"]
            })

    def test_missing_training_states_do_not_broaden_a_known_state_profile(self):
        with self.assertWarnsRegex(RuntimeWarning, "CC_1.CurrentState: 1"):
            model = self.train(pd.DataFrame({
                "CC_1.CurrentState": pd.array([True, None, False], dtype="boolean"),
                "CC_1.CurrentSpeed": [10., 999., 0.],
            }))
        self.assertEqual(model["CC_1.CurrentSpeed"]["states"]["True"]["upper"], 10.)


if __name__ == "__main__":
    unittest.main()
