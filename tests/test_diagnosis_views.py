import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dashboard.features.case_based import CaseBasedDX
from dashboard.features.structural_knowledge import StructKnowledgeDX


class DiagnosisViewTests(unittest.TestCase):
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
