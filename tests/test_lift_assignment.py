import json
import unittest
from unittest.mock import patch

from scripts.prepare_data import flatten_sample


class LiftAssignmentTests(unittest.TestCase):
    @patch("scripts.prepare_data.LIFT_ORDER", ["CLT_28", "CLT_15"])
    def test_assigns_all_variables_in_configured_order(self):
        values = [
            {"CLT_15": {"CLT_15.CurrentSpeed": "4"}},
            {"CLT_28": {"CLT_28.CurrentSpeed": "3"}},
            {"Lift": {"Lift.CurrentSpeed": "1,5", "Lift.MotorFault": "True"}},
            {"Lift": {"Lift.CurrentSpeed": "2,5", "Lift.MotorFault": "False"}},
        ]
        for exported_values in (values, json.dumps(values)):
            with self.subTest(encoded=isinstance(exported_values, str)):
                row = flatten_sample({"simulationTime": 1, "values": exported_values})
                self.assertEqual(row, {
                    "simulationTime": 1,
                    "CLT_28.CurrentSpeed": 3.0,
                    "CLT_15.CurrentSpeed": 4.0,
                    "CLT_28.Lift.CurrentSpeed": 1.5,
                    "CLT_28.Lift.MotorFault": True,
                    "CLT_15.Lift.CurrentSpeed": 2.5,
                    "CLT_15.Lift.MotorFault": False,
                })

    def test_rejects_ambiguous_owner_counts(self):
        for names in ([], ["CLT_15", "CLT_28"], ["CLT_15", "CLT_15"], ["RC_15"]):
            with self.subTest(names=names), patch("scripts.prepare_data.LIFT_ORDER", names):
                count = max(1, len(names)) if len(set(names)) != len(names) else 1
                values = [{"Lift": {"Lift.CurrentSpeed": "1"}} for _ in range(count)]
                with self.assertRaisesRegex(ValueError, "Cannot assign"):
                    flatten_sample({"values": values})

    def test_preserves_already_qualified_lifts(self):
        row = flatten_sample({"values": [
            {"CLT_15": {"CLT_15.Lift.CurrentSpeed": "1,5"}},
        ]})
        self.assertEqual(row["CLT_15.Lift.CurrentSpeed"], 1.5)

    @patch("scripts.prepare_data.LIFT_ORDER", ["CLT_15"])
    def test_rejects_colliding_lift_variables(self):
        with self.assertRaisesRegex(ValueError, "Duplicate Lift"):
            flatten_sample({"values": [
                {"CLT_15": {"CLT_15.Lift.CurrentSpeed": "2"}},
                {"Lift": {"Lift.CurrentSpeed": "1"}},
            ]})


if __name__ == "__main__":
    unittest.main()
