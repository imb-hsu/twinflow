import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_data import (
    COMPONENT_TYPE_KNOWLEDGE_PATH,
    normalize_dataframe,
    split_dataframe,
)


class TypedExportTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = json.loads(COMPONENT_TYPE_KNOWLEDGE_PATH.read_text())

    def convert(self, df):
        with contextlib.redirect_stdout(io.StringIO()):
            return normalize_dataframe(df, self.knowledge)

    def test_vectors_categories_and_missing_values(self):
        df = pd.DataFrame({
            "AMR_1.CurrentDirection": ["<1,5. -2. 3E-2>", None],
            "PR_7.X.CurrentForce": ["<1.234,5. 0. -0>", "<?. 0. 1>"],
            "AMR_1.CurrentState": ["Turning", None],
            "CC_1.CurrentDirection": ["Reverse", None],
            "CC_1.PeWrongCalibration": ["", "?"],
            "CC_1.MotorFault": [True, None],
            "CC_1.Count": [42, 43],
        })
        result = self.convert(df)
        self.assertNotIn("AMR_1.CurrentDirection", result)
        self.assertEqual(result["AMR_1.CurrentDirection.X"].iloc[0], 1.5)
        self.assertEqual(result["AMR_1.CurrentDirection.Y"].iloc[0], -2)
        self.assertEqual(result["AMR_1.CurrentDirection.Z"].iloc[0], 0.03)
        self.assertEqual(result["PR_7.X.CurrentForce.X"].iloc[0], 1234.5)
        self.assertTrue(pd.isna(result["PR_7.X.CurrentForce.X"].iloc[1]))
        self.assertEqual(result["AMR_1.CurrentState"].iloc[0], "Turning")
        self.assertEqual(result["CC_1.CurrentDirection"].iloc[0], "Reverse")
        self.assertTrue(result["CC_1.PeWrongCalibration"].isna().all())
        self.assertTrue(pd.api.types.is_bool_dtype(result["CC_1.MotorFault"].dtype))
        self.assertTrue(result["CC_1.MotorFault"].iloc[0])
        self.assertTrue(pd.isna(result["CC_1.MotorFault"].iloc[1]))
        self.assertTrue(pd.isna(result["AMR_1.CurrentState"].iloc[1]))
        self.assertTrue(pd.api.types.is_integer_dtype(result["CC_1.Count"].dtype))

    def test_invalid_vectors_and_collisions_raise(self):
        for columns in (
            {"AMR_1.CurrentDirection": ["<1. 2>"]},
            {"AMR_1.CurrentDirection": ["<1. 2. 3>"], "AMR_1.CurrentDirection.X": [4]},
        ):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                self.convert(pd.DataFrame(columns))

    def test_parquet_outputs_preserve_types(self):
        df = pd.DataFrame({
            "AMR_1.CurrentState": ["Moving", None],
            "AMR_1.TargetSpeed": [1.0, 2.0],
            "AMR_1.MotorFault": [True, None],
        }, index=pd.Index([0, 1], name="simulationTime"))
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataframe(df, directory)
            for group in ("measurements", "parameters", "faults"):
                result = pd.read_parquet(Path(directory) / f"{group}.parquet")
                self.assertEqual(result.index.name, "simulationTime")
                if group == "measurements":
                    self.assertEqual(result["AMR_1.CurrentState"].iloc[0], "Moving")
                    self.assertTrue(pd.isna(result["AMR_1.CurrentState"].iloc[1]))
                    self.assertIsInstance(result["AMR_1.CurrentState"].dtype, pd.StringDtype)
                elif group == "faults":
                    self.assertTrue(pd.api.types.is_bool_dtype(result["AMR_1.MotorFault"].dtype))
                    self.assertTrue(pd.isna(result["AMR_1.MotorFault"].iloc[1]))
                else:
                    self.assertTrue(pd.api.types.is_numeric_dtype(result["AMR_1.TargetSpeed"].dtype))


if __name__ == "__main__":
    unittest.main()
