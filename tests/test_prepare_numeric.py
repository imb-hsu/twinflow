import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import zstandard as zstd

from scripts.prepare_data import (
    COMPONENT_TYPE_KNOWLEDGE_PATH,
    main,
    normalize_dataframe,
    split_dataframe,
    update_faults_parquet,
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
                    self.assertFalse(result["AMR_1.MotorFault"].iloc[1])
                else:
                    self.assertTrue(pd.api.types.is_numeric_dtype(result["AMR_1.TargetSpeed"].dtype))

    def test_faults_are_zero_only_while_their_own_component_is_inactive(self):
        df = pd.DataFrame({
            "CC_1.CurrentState": [False, True, False, None],
            "CC_1.MotorFault": [True, True, None, True],
            "CC_1.SpeedCalibrationFault": [2.5, -0.5, None, 1.5],
            "CC_1.TargetSpeed": [3.0] * 4,
            "CLT_15.CurrentState": [False, True, False, True],
            "CLT_15.MotorFault": [True] * 4,
            "CLT_15.Lift.CurrentState": [True, False, True, False],
            "CLT_15.Lift.MotorFault": [True] * 4,
            "AMR_1.CurrentState": ["Idle", "Moving", "Turning", None],
            "AMR_1.MotorFault": [True] * 4,
            "AMR_1.AMP.CurrentState": [True, False, False, True],
            "AMR_1.AMP.WearFault": [0.4] * 4,
            "RB_1.MotorFault": [True] * 4,
        }, index=pd.Index([0.5, 1.0, 1.5, 2.0], name="simulationTime"))
        original = df.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataframe(df, directory)
            faults = pd.read_parquet(Path(directory) / "faults.parquet")
            measurements = pd.read_parquet(Path(directory) / "measurements.parquet")
            parameters = pd.read_parquet(Path(directory) / "parameters.parquet")

        self.assertEqual(faults["CC_1.MotorFault"].tolist(), [False, True, False, True])
        self.assertEqual(faults["CC_1.SpeedCalibrationFault"].tolist(), [0, -0.5, 0, 1.5])
        self.assertEqual(faults["CLT_15.MotorFault"].tolist(), [False, True, False, True])
        self.assertEqual(faults["CLT_15.Lift.MotorFault"].tolist(), [True, False, True, False])
        self.assertEqual(faults["AMR_1.MotorFault"].tolist(), [False, True, True, True])
        self.assertEqual(faults["AMR_1.AMP.WearFault"].tolist(), [0.4, 0, 0, 0.4])
        self.assertEqual(faults["RB_1.MotorFault"].tolist(), [True] * 4)
        self.assertTrue(pd.api.types.is_bool_dtype(faults["CC_1.MotorFault"].dtype))
        pd.testing.assert_index_equal(faults.index, df.index)
        pd.testing.assert_series_equal(
            measurements["CC_1.CurrentState"], self.convert(df)["CC_1.CurrentState"]
        )
        pd.testing.assert_series_equal(parameters["CC_1.TargetSpeed"], df["CC_1.TargetSpeed"])
        pd.testing.assert_frame_equal(df, original)

    def test_pr_faults_are_zero_when_all_axes_are_inactive(self):
        df = pd.DataFrame({
            "PR_7.X.CurrentState": [False, True, False, None],
            "PR_7.Y.CurrentState": [False, False, True, False],
            "PR_7.Z.CurrentState": [False, False, False, False],
            "PR_7.MotorFault": [True] * 4,
            "PR_7.WearFault": [0.3] * 4,
            "PR_118.X.CurrentState": [False] * 4,
            "PR_118.MotorFault": [True] * 4,
        })
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataframe(df, directory)
            faults = pd.read_parquet(Path(directory) / "faults.parquet")

        self.assertEqual(faults["PR_7.MotorFault"].tolist(), [False, True, True, True])
        self.assertEqual(faults["PR_7.WearFault"].tolist(), [0, 0.3, 0.3, 0.3])
        self.assertEqual(faults["PR_118.MotorFault"].tolist(), [True] * 4)

    def test_nominal_damping_is_zero_regardless_of_component_activity(self):
        df = pd.DataFrame({
            "CC_1.CurrentState": [True, True, True, None, False, True],
            "CC_1.IncreasedDampingFault": [0.1, 0.2, None, 0.1, 0.2, 0.1000001],
            "CC_1.WearFault": [0.1] * 6,
            "AMR_1.AMP.IncreasedDampingFault": [0.1] * 6,
            "CLT_15.Lift.IncreasedDampingFault": [0.1] * 6,
        }, index=pd.Index(range(6), name="simulationTime"))
        original = df.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataframe(df, directory)
            faults = pd.read_parquet(Path(directory) / "faults.parquet")

        pd.testing.assert_series_equal(faults["CC_1.IncreasedDampingFault"], pd.Series(
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.1000001],
            index=df.index, name="CC_1.IncreasedDampingFault",
        ))
        self.assertEqual(faults["CC_1.WearFault"].tolist(), [0.1, 0.1, 0.1, 0.1, 0, 0.1])
        for column in ("AMR_1.AMP.IncreasedDampingFault", "CLT_15.Lift.IncreasedDampingFault"):
            self.assertEqual(faults[column].tolist(), [0.0] * 6)
        pd.testing.assert_frame_equal(df, original)

    def test_missing_faults_are_zero_for_active_unknown_and_unexported_states(self):
        df = pd.DataFrame({
            "CC_1.CurrentState": pd.array([True, False, None], dtype="boolean"),
            "CC_1.MotorFault": pd.array([None, None, None], dtype="boolean"),
            "CC_1.PeWrongCalibration": pd.array([None, None, None], dtype="Float64"),
            "RB_1.MotorFault": pd.array([None, True, None], dtype="boolean"),
            "RB_1.SpeedCalibrationFault": [None, 0.5, None],
        })
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataframe(df, directory)
            faults = pd.read_parquet(Path(directory) / "faults.parquet")
            measurements = pd.read_parquet(Path(directory) / "measurements.parquet")
        self.assertFalse(faults.isna().any().any())
        self.assertEqual(faults["CC_1.MotorFault"].tolist(), [False] * 3)
        self.assertEqual(faults["CC_1.PeWrongCalibration"].tolist(), [0.0] * 3)
        self.assertEqual(faults["RB_1.MotorFault"].tolist(), [False, True, False])
        self.assertEqual(faults["RB_1.SpeedCalibrationFault"].tolist(), [0.0, 0.5, 0.0])
        self.assertEqual(faults["CC_1.MotorFault"].dtype, pd.BooleanDtype())
        self.assertEqual(faults["CC_1.PeWrongCalibration"].dtype, pd.Float64Dtype())
        pd.testing.assert_series_equal(measurements["CC_1.CurrentState"], df["CC_1.CurrentState"])


class SavedFaultPostprocessingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.training = Path(temporary.name) / "training"
        self.test = Path(temporary.name) / "test"
        self.training.mkdir()
        self.test.mkdir()
        self.output = self.training / "existing"
        self.output.mkdir()
        index = pd.Index([0.0, 1.0], name="simulationTime")
        pd.DataFrame({"CC_1.CurrentState": pd.array([False, True], dtype="boolean")}, index=index).to_parquet(
            self.output / "measurements.parquet"
        )
        pd.DataFrame({
            "CC_1.MotorFault": pd.array([True, None], dtype="boolean"),
            "CC_1.WearFault": [float("nan"), 0.3],
            "CC_1.IncreasedDampingFault": [0.1, 0.1],
            "RB_1.MotorFault": pd.array([None, True], dtype="boolean"),
            "RB_1.SpeedCalibrationFault": [float("nan"), float("nan")],
        }, index=index).to_parquet(self.output / "faults.parquet")
        pd.DataFrame({"CC_1.TargetSpeed": [2.0, 2.0]}, index=index).to_parquet(
            self.output / "parameters.parquet"
        )
        self.directories = patch("scripts.prepare_data.DIRS_TO_TRANSFORM", [self.training, self.test])
        self.directories.start()
        self.addCleanup(self.directories.stop)

    def assert_existing_faults_updated(self):
        faults = pd.read_parquet(self.output / "faults.parquet")
        self.assertFalse(faults["CC_1.MotorFault"].iloc[0])
        self.assertFalse(faults["CC_1.MotorFault"].iloc[1])
        self.assertEqual(faults["CC_1.WearFault"].tolist(), [0.0, 0.3])
        self.assertEqual(faults["CC_1.IncreasedDampingFault"].tolist(), [0.0, 0.0])
        self.assertEqual(faults["RB_1.MotorFault"].tolist(), [False, True])
        self.assertEqual(faults["RB_1.SpeedCalibrationFault"].tolist(), [0.0, 0.0])
        self.assertFalse(faults.isna().any().any())
        self.assertTrue(pd.api.types.is_bool_dtype(faults["CC_1.MotorFault"].dtype))
        self.assertEqual(faults.index.name, "simulationTime")

    def test_main_updates_existing_faults_after_all_raw_exports_are_prepared(self):
        original_files = {
            name: (self.output / name).read_bytes()
            for name in ("measurements.parquet", "parameters.parquet")
        }
        samples = [
            {"simulationTime": time, "values": [{"CC_2": {
                "CC_2.CurrentState": state, "CC_2.MotorFault": "True",
            }}]}
            for time, state in [(0.0, "False"), (1.0, "True")]
        ]
        archive = self.test / "new.json.zst"
        archive.write_bytes(zstd.ZstdCompressor().compress(
            "\n".join(json.dumps(sample) for sample in samples).encode("utf-8")
        ))

        def update_after_preparation(directory):
            self.assertTrue((self.test / "new" / "measurements.parquet").is_file())
            update_faults_parquet(directory)

        with contextlib.redirect_stdout(io.StringIO()), patch(
            "scripts.prepare_data.update_faults_parquet", side_effect=update_after_preparation
        ) as update:
            main()

        self.assertEqual(update.call_count, 2)
        self.assert_existing_faults_updated()
        new_faults = pd.read_parquet(self.test / "new" / "faults.parquet")
        self.assertEqual(new_faults["CC_2.MotorFault"].tolist(), [False, True])
        for name, content in original_files.items():
            self.assertEqual((self.output / name).read_bytes(), content)

    def test_faults_only_skips_raw_conversion_and_can_be_repeated(self):
        (self.test / "invalid.json.zst").write_bytes(b"not a compressed export")
        with contextlib.redirect_stdout(io.StringIO()), patch("scripts.prepare_data.read_json_zst") as read:
            main(faults_only=True)
            first = pd.read_parquet(self.output / "faults.parquet")
            main(faults_only=True)
        read.assert_not_called()
        self.assert_existing_faults_updated()
        pd.testing.assert_frame_equal(first, pd.read_parquet(self.output / "faults.parquet"))

    def test_mismatched_timestamps_do_not_overwrite_faults(self):
        path = self.output / "faults.parquet"
        original = path.read_bytes()
        measurements_path = self.output / "measurements.parquet"
        measurements = pd.read_parquet(measurements_path)
        measurements.index = pd.Index([1.0, 2.0], name="simulationTime")
        measurements.to_parquet(measurements_path)
        with self.assertRaisesRegex(ValueError, "identical simulationTime indexes"):
            update_faults_parquet(self.output)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
