import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
from dash import Dash

from dashboard.features import data_plots


class DataPlotsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.data = root / "data" / "test" / "example"
        self.data.mkdir(parents=True)
        self.scenarios = root / "scenarios"
        (self.scenarios / "test").mkdir(parents=True)
        self.scenario = self.scenarios / "test" / "example.json"
        self.scenario.write_text('{"scenario": {}, "actions": []}')
        index = pd.Index(range(6001), name="simulationTime")
        pd.DataFrame({"speed": 1.0}, index=index).to_parquet(self.data / "measurements.parquet")
        pd.DataFrame({"limit": 2.0}, index=index).to_parquet(self.data / "parameters.parquet")
        faults = pd.DataFrame({"zero": 0.0, "missing": float("nan"), "pulse": 0.0,
                               "negative": -0.5, "boolean": False}, index=index)
        faults.loc[1, "pulse"] = 2.0  # Would be missed by stride-2 downsampling.
        faults.loc[3, "boolean"] = True
        faults.to_parquet(self.data / "faults.parquet")
        self.enterContext(patch.object(data_plots, "DATA_PATH", root / "data"))
        self.enterContext(patch.object(data_plots, "SCENARIOS_PATH", self.scenarios))
        self.feature = data_plots.DataPlots()

    def test_fault_selection_preserves_short_spikes(self):
        faults = self.feature._read_active_faults("test", "example")
        self.assertEqual(set(faults.columns), {"pulse", "negative", "boolean"})
        self.assertEqual(faults.loc[1, "pulse"], 2)
        self.assertEqual(faults.loc[3, "boolean"], 1)
        self.assertTrue({0, 1, 2, 3, 4, 6000}.issubset(faults.index))
        figure, _ = self.feature._plot_figure("test", "example", "measurements", [])
        traces = [trace for trace in figure.data if trace.yaxis == "y3"]
        self.assertEqual({trace.name for trace in traces}, set(faults.columns))
        self.assertEqual(figure.layout.xaxis5.matches, "x")

    def test_measurement_sample_limit_is_configurable(self):
        for limit, expected in ((1000, 858), (None, 6001), (10000, 6001)):
            with self.subTest(limit=limit):
                self.feature.config["max_plot_points"]["value"] = limit
                frame = self.feature._read_plot_data("test", "example", "measurements", ["speed"])
                self.assertEqual(len(frame), expected)
                self.assertEqual(frame.attrs["total_samples"], 6001)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid), patch.object(data_plots.DataPlots, "max_plot_points", invalid):
                with self.assertRaisesRegex(ValueError, "max_plot_points"):
                    data_plots.DataPlots()

    def test_empty_and_missing_faults_are_explained(self):
        pd.DataFrame({"zero": [0, 0]}).to_parquet(self.data / "faults.parquet")
        figure, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"])
        self.assertIn("No nonzero faults", next(t for t in figure.data if t.yaxis == "y3").text[0])
        (self.data / "faults.parquet").unlink()
        figure, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"])
        self.assertIn("Could not load faults", next(t for t in figure.data if t.yaxis == "y3").text[0])

    def test_increased_damping_fault_treats_0_1_as_normal(self):
        index = pd.Index(range(4), name="simulationTime")
        pd.DataFrame({"AMP_1.IncreasedDampingFault": [0.1, 0.1, 0.1, 0.1]}, index=index).to_parquet(
            self.data / "faults.parquet"
        )
        self.assertNotIn("AMP_1.IncreasedDampingFault", self.feature._read_active_faults("test", "example").columns)

        pd.DataFrame({"AMP_1.IncreasedDampingFault": [0.0, 0.1, 0.5, 0.0]}, index=index).to_parquet(
            self.data / "faults.parquet"
        )
        self.assertIn("AMP_1.IncreasedDampingFault", self.feature._read_active_faults("test", "example").columns)

    def test_parameters_and_measurements_follow_faults(self):
        default, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"])
        self.assertFalse(any(t.legendgroup == "parameters" for t in default.data))
        self.assertIn("Select parameter signals", next(t for t in default.data if t.yaxis == "y4").text[0])
        figure, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"], ["limit"])
        parameter = next(trace for trace in figure.data if trace.name == "limit")
        measurement = next(trace for trace in figure.data if trace.name == "speed")
        self.assertEqual(parameter.yaxis, "y4")
        self.assertEqual(list(parameter.x), [0, 6000])
        self.assertEqual(list(parameter.y), [2, 2])
        self.assertEqual(measurement.yaxis, "y5")
        self.assertEqual([a.text for a in figure.layout.annotations], [
            "Scenario stateflow", "PalletCreator events", "Faults with nonzero values",
            "Parameters", "measurements.parquet data",
        ])
        self.assertTrue(all(
            (getattr(figure.layout, f"yaxis{i}", None) is None or getattr(figure.layout, f"yaxis{i}").title.text == "")
            for i in range(1, 6)
        ))
        (self.data / "parameters.parquet").unlink()
        figure, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"], ["limit"])
        self.assertIn("Could not load parameters", next(t for t in figure.data if t.yaxis == "y4").text[0])

    def test_download_contains_exactly_four_original_files(self):
        result = self.feature._download_bundle("test/example")
        self.assertEqual(result["filename"], "test_example.zip")
        expected = [self.scenario] + [self.data / name for name in data_plots.TABLES.values()]
        with ZipFile(io.BytesIO(base64.b64decode(result["content"]))) as archive:
            self.assertEqual(set(archive.namelist()), {p.name for p in expected})
            for path in expected:
                self.assertEqual(archive.read(path.name), path.read_bytes())
        with self.assertRaises(ValueError):
            self.feature._download_bundle("test/../../outside")
        (self.data / "parameters.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "parameters.parquet"):
            self.feature._download_bundle("test/example")

    def test_parameter_validation_checks_unselected_signals(self):
        for values in ([1.0, 2.0], [None, None]):
            with self.subTest(values=values):
                pd.DataFrame({"changing_limit": values}).to_parquet(self.data / "parameters.parquet")
                with self.assertRaisesRegex(ValueError, "changing_limit"):
                    self.feature._read_constant_parameters("test", "example")
                figure, _ = self.feature._plot_figure("test", "example", "measurements", ["speed"])
                message = next(trace for trace in figure.data if trace.yaxis == "y4").text[0]
                self.assertIn("Parameters must be constant", message)
                self.assertIn("changing_limit", message)

    def test_callbacks_and_layout_serialize(self):
        app = Dash(__name__)
        from dash import html
        app.layout = html.Div([
            self.feature.topbar_controls(), self.feature.layout(None, None, None, None)
        ])
        self.feature.register_callbacks(app, None)
        self.feature.register_callbacks(app, None)
        self.assertEqual(len(app.callback_map), 5)
        client = app.server.test_client()
        self.assertEqual(client.get("/_dash-layout").status_code, 200)
        self.assertEqual(client.get("/_dash-dependencies").status_code, 200)
        key = next(key for key in app.callback_map if "download-status.children" in key)
        response = client.post("/_dash-update-component", json={
            "output": key,
            "outputs": [{"id": self.feature._download_id, "property": "data"},
                        {"id": self.feature._download_status_id, "property": "children"}],
            "inputs": [{"id": self.feature._download_button_id, "property": "n_clicks", "value": 1}],
            "state": [{"id": self.feature._file_selector_id, "property": "value", "value": "test/example"}],
            "changedPropIds": [self.feature._download_button_id + ".n_clicks"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["response"][self.feature._download_id]["data"]["filename"], "test_example.zip")

        # Applying SelfX's Configure modal updates config before emitting is_open=False.
        self.feature.config["max_plot_points"]["value"] = 1000
        key = next(key for key in app.callback_map if self.feature._graph_id + ".figure" in key)
        modal_id = data_plots.construct_id("system", self.feature.feature_name(), "modal")
        response = client.post("/_dash-update-component", json={
            "output": key,
            "outputs": [{"id": self.feature._graph_id, "property": "figure"},
                        {"id": self.feature._summary_id, "property": "children"}],
            "inputs": [
                {"id": self.feature._file_selector_id, "property": "value", "value": "test/example"},
                {"id": self.feature._columns_selector_id, "property": "value", "value": ["speed"]},
                {"id": self.feature._parameters_selector_id, "property": "value", "value": []},
                {"id": modal_id, "property": "is_open", "value": False},
            ],
            "state": [],
            "changedPropIds": [modal_id + ".is_open"],
        })
        self.assertEqual(response.status_code, 200)
        figure = response.json["response"][self.feature._graph_id]["figure"]
        self.assertEqual(len(next(trace for trace in figure["data"] if trace.get("name") == "speed")["x"]), 858)


if __name__ == "__main__":
    unittest.main()
