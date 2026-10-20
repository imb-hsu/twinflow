import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dashboard.features.case_based import CaseBasedDX
from dashboard.features.structural_knowledge import StructKnowledgeDX


class DiagnosisViewTests(unittest.TestCase):
    def test_structural_backend_and_dashboard_choose_nominal_only_thresholds(self):
        from methods.dx import structural_knowledge
        from dashboard.features import structural_knowledge as view
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "structural_knowledge.json"
            model_path.write_text(json.dumps({"component_type_by_id": {"CC_1": "CC"},
                                              "faults_by_type": {"CC": ["MotorFault"]}}))
            nominal = {"CC_1.CurrentSpeed": {"type": "numeric", "lower": 0., "upper": 1.}}
            (root / "range_monitoring_0pct.json").write_text(json.dumps(nominal))
            (root / "range_monitoring.json").write_text(json.dumps({
                "CC_1.CurrentSpeed": {"type": "numeric", "lower": 0., "upper": 10.}
            }))
            with patch.object(view, "MODELS_PATH", root):
                loaded = StructKnowledgeDX()._load_model(model_path)
            self.assertEqual(loaded["thresholds"], nominal)
            with patch.object(structural_knowledge, "MODEL_PATH", model_path), patch.object(
                structural_knowledge, "THRESHOLDS_PATH", root / structural_knowledge.THRESHOLDS_PATH.name
            ):
                prediction = structural_knowledge.predict(pd.DataFrame({"CC_1.CurrentSpeed": [2.]}))
            self.assertEqual(tuple(prediction.iloc[0]), ("CC_1", "MotorFault"))

    def test_structural_view_uses_component_state_profiles(self):
        frame = pd.DataFrame({"CC_1.CurrentState": [False, True], "CC_1.CurrentSpeed": [0., 0.]})
        faults = pd.DataFrame({"CC_1.MotorFault": [False, True]})
        model = {"thresholds": {"CC_1.CurrentSpeed": {
            "type": "numeric", "state_column": "CC_1.CurrentState", "states": {
                "False": {"type": "numeric", "lower": 0., "upper": 0.},
                "True": {"type": "numeric", "lower": 9., "upper": 11.},
            },
        }}, "component_type_by_id": {"CC_1": "CC"}, "area_by_id": {},
            "faults_by_type": {"CC": ["MotorFault"]}}
        feature = StructKnowledgeDX()
        with patch("dashboard.features.structural_knowledge.load_table", side_effect=[frame, faults]):
            figure, _ = feature._analyze("test", "example", model)
        prediction = next(trace for trace in figure.data if trace.name == "Predicted diagnosis")
        self.assertEqual(list(prediction.y), ["CC_1.MotorFault"])

    def test_views_only_analyze_fault_active_samples(self):
        frame = pd.DataFrame({"Motor.speed": [99., 2., 3., 99.]})
        faults = pd.DataFrame({"Motor.Fault": [0., 1., 1., 0.],
                               "Motor.IncreasedDampingFault": [0.1] * 4})
        from sklearn.neighbors import NearestNeighbors
        cases = {"columns": ["Motor.speed"], "scaler": {"mean": [0.], "scale": [1.]},
                 "cases": [{"fault_label": "Motor.Fault"}],
                 "_neighbors": NearestNeighbors(n_neighbors=1).fit(np.array([[2.]]))}
        structural = {"thresholds": {"Motor.speed": {"lower": 0., "upper": 1.}},
                      "component_type_by_id": {"Motor": "Motor"}, "area_by_id": {},
                      "faults_by_type": {"Motor": ["Fault"]}}
        for module, cls, model in [("case_based", CaseBasedDX, cases),
                                   ("structural_knowledge", StructKnowledgeDX, structural)]:
            feature = cls()
            with patch(f"dashboard.features.{module}.load_table", side_effect=[frame, faults]):
                figure, matrix = feature._analyze("test", "AMR_30pct_faults", model)
            prediction = next(trace for trace in figure.data if trace.name == "Predicted diagnosis")
            self.assertEqual(list(prediction.x), [2])
            self.assertEqual(list(prediction.y), ["Motor.Fault"])
            self.assertEqual(figure.layout.yaxis.categoryarray, ("Target", "Predicted"))
            self.assertEqual(matrix.children.children[2].children[0].children[1].children, 1)
            figure.to_json()
            for labels, message in [(None, "unavailable"), (faults * 0, "No faulty intervals")]:
                with patch(f"dashboard.features.{module}.load_table", side_effect=[frame, labels]):
                    figure, summary = feature._analyze("test", "AMR_30pct_faults", model)
                self.assertFalse(figure.data)
                self.assertIn(message, summary)
