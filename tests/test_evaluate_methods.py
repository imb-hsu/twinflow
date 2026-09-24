import contextlib
import io
import importlib.util
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from methods.ad import autoencoder, range_monitoring
from methods.dx import case_based, structural_knowledge
from scripts import evaluate_methods as evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'models'
        self.models.mkdir()
        for module in (range_monitoring, autoencoder, case_based, structural_knowledge):
            self.enterContext(patch.object(module, 'MODEL_PATH', self.models / (module.__name__.rsplit('.', 1)[-1] + '.json')))
        self.enterContext(patch.object(structural_knowledge, 'THRESHOLDS_PATH', self.models / 'range_monitoring.json'))
        training = self.root / 'data/training/0_0pct_test'
        training.mkdir(parents=True)
        (training / 'measurements.parquet').touch()

    def test_training_saves_model_without_test_data_or_metrics(self):
        training = self.root / 'data/training/0_0pct_test'
        pd.DataFrame({'speed': [0., 1.], 'state': ['idle', 'run']}).to_parquet(
            training / 'measurements.parquet')
        with patch.object(range_monitoring, 'REPO_ROOT', self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            range_monitoring.train()
        model = json.loads((self.models / 'range_monitoring.json').read_text())
        self.assertEqual(model['speed'], {'type': 'numeric', 'lower': 0., 'upper': 1.})
        self.assertEqual(model['state']['categories'], ['idle', 'run'])
        predictions = range_monitoring.predict(
            pd.DataFrame({'speed': [0., 2., 0., np.nan],
                          'state': ['idle', 'run', 'unknown', 'idle']}))
        self.assertEqual(predictions.tolist(), [False, True, True, True])

    def test_saved_detectors_use_the_same_labels_and_metrics(self):
        (self.models / 'range_monitoring.json').write_text(json.dumps({
            'speed': {'type': 'numeric', 'lower': -1., 'upper': 1.}}))
        (self.models / 'autoencoder.json').write_text(json.dumps({
            'columns': ['speed'], 'scaler': {'mean': [0.], 'scale': [1.]},
            'layers': [{'weights': [[0.]], 'bias': [0.]}], 'threshold': 1.,
        }))
        datasets = {}
        for name in ('AMR_30pct_faults', 'WE_30pct_faults'):
            directory = self.root / name
            directory.mkdir()
            pd.DataFrame({'speed': [0., 2., 0.]}).to_parquet(directory / 'measurements.parquet')
            pd.DataFrame({'MotorFault': [False, True, False],
                          'IncreasedDampingFault': [0.1] * 3}).to_parquet(directory / 'faults.parquet')
            datasets[name] = {table: pd.read_parquet(directory / (table + '.parquet')) for table in ('measurements', 'faults')}
        with contextlib.nullcontext(), contextlib.redirect_stdout(io.StringIO()):
            result = evaluate.evaluate_anomaly_detection(datasets)
        self.assertEqual(result['Range Monitoring'], result['Vanilla Autoencoder'])
        scores = result['Range Monitoring']
        self.assertEqual((scores['tp'], scores['tn'], scores['fp'], scores['fn']), (2, 4, 0, 0))
        self.assertEqual(scores['F1_comp'], 1.)
        self.assertEqual(len(scores['scenarios']), 2)

    def test_saved_diagnosis_models_use_method_predict_functions(self):
        scaler = StandardScaler().fit(np.array([[0.], [2.]]))
        model = {
            'columns': ['CC_1.speed'], 'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
            'cases': [{'vector': scaler.transform([[2.]])[0].tolist(),
                       'component': 'CC_1', 'fault_type': 'MotorFault'}],
        }
        (self.models / 'case_based.json').write_text(json.dumps(model))
        (self.models / 'structural_knowledge.json').write_text(json.dumps({
            'component_type_by_id': {'CC_1': 'CC'},
            'faults_by_type': {'CC': ['MotorFault']}}))
        (self.models / 'range_monitoring.json').write_text(json.dumps({
            'CC_1.speed': {'type': 'numeric', 'lower': 0., 'upper': 1.}}))
        directory = self.root / 'scenario'
        directory.mkdir()
        pd.DataFrame({'CC_1.speed': [0., 2., 2., 0.]}).to_parquet(directory / 'measurements.parquet')
        pd.DataFrame({'CC_1.MotorFault': [False, True, True, False]}).to_parquet(directory / 'faults.parquet')
        with contextlib.nullcontext(), \
                patch.object(case_based, 'predict', wraps=case_based.predict) as cb, \
                patch.object(structural_knowledge, 'predict', wraps=structural_knowledge.predict) as sk, \
                contextlib.redirect_stdout(io.StringIO()):
            result = evaluate.evaluate_diagnosis({'scenario': {table: pd.read_parquet(directory / (table + '.parquet')) for table in ('measurements', 'faults')}})
        cb.assert_called_once()
        sk.assert_called_once()
        pd.testing.assert_frame_equal(cb.call_args.args[0], sk.call_args.args[0])
        self.assertEqual(cb.call_args.args[0].index.tolist(), [2])
        for scores in result.values():
            self.assertEqual(scores, {'Loc_BA': 1., 'Loc_F1': 1., 'Fault_BA': 1., 'Fault_F1': 1.})

    def test_diagnosis_endpoints_include_singletons_overlaps_and_final_row(self):
        labels = pd.DataFrame({'A.Fault': [True, False, True, True],
                               'B.Fault': [False, True, True, True]})
        self.assertEqual(evaluate.diagnosis_events(labels),
                         [(0, 'A.Fault'), (3, 'A.Fault'), (3, 'B.Fault')])

    def test_legacy_nominal_cases_are_excluded(self):
        model = {
            'columns': ['speed'], 'scaler': {'mean': [0.], 'scale': [1.]},
            'cases': [
                {'vector': [0.], 'fault_label': 'nominal', 'component': 'nominal', 'fault_type': 'nominal'},
                {'vector': [2.], 'fault_label': 'CC_1.MotorFault', 'component': 'CC_1', 'fault_type': 'MotorFault'},
            ],
        }
        case_based.MODEL_PATH.write_text(json.dumps(model))
        result = case_based.predict(pd.DataFrame({'speed': [0.]}))
        self.assertEqual(result['fault_type'].tolist(), ['MotorFault'])
        from dashboard.features.case_based import CaseBasedDX
        artifact = CaseBasedDX()._load_model(case_based.MODEL_PATH)
        self.assertEqual([case['fault_label'] for case in artifact['cases']], ['CC_1.MotorFault'])

    def test_autoencoder_training_json_preserves_predictions(self):
        from sklearn.neural_network import MLPRegressor
        measurements = pd.DataFrame({'x': np.linspace(-2, 2, 40),
                                     'y': np.sin(np.linspace(-2, 2, 40)),
                                     'enabled': [True, False] * 20,
                                     'state': ['idle'] * 20 + ['running'] * 20})
        fitted = MLPRegressor(hidden_layer_sizes=(3,), activation='relu',
                              solver='lbfgs', max_iter=1000, random_state=1)
        measurements.iloc[:20].to_parquet(self.root / 'data/training/0_0pct_test/measurements.parquet')
        faulty = self.root / 'data/training/10_0pct_test'
        faulty.mkdir()
        measurements.iloc[20:].to_parquet(faulty / 'measurements.parquet')
        # Test data must not influence the trained model.
        test = self.root / 'data/test/example'
        test.mkdir(parents=True)
        measurements.assign(x=measurements['x'] * 100).to_parquet(test / 'measurements.parquet')
        with patch.object(autoencoder, 'REPO_ROOT', self.root), \
                patch.object(autoencoder, 'ROW_STRIDE', 1), \
                patch.object(autoencoder, 'MLPRegressor', return_value=fitted), \
                contextlib.redirect_stdout(io.StringIO()):
            autoencoder.train()
        model = json.loads((self.models / 'autoencoder.json').read_text())
        self.assertEqual(model['encoding']['enabled']['type'], 'boolean')
        self.assertEqual(model['encoding']['state']['categories'], ['idle', 'running'])
        encoded = autoencoder.encode_measurements(measurements, model)
        self.assertEqual(encoded.shape, (40, 5))
        np.testing.assert_allclose(model['scaler']['mean'], encoded.mean(axis=0), atol=1e-12)
        probe = measurements.assign(x=measurements['x'] * 3, y=measurements['y'] * 3)
        scaled = (autoencoder.encode_measurements(probe, model) - model['scaler']['mean']) / model['scaler']['scale']
        expected_errors = np.mean((scaled - fitted.predict(scaled)) ** 2, axis=1)
        np.testing.assert_array_equal(autoencoder.predict(probe), expected_errors > model['threshold'])
        # Dashboard reconstruction must also load the same JSON artifact.
        from dashboard.features.autoencoder import AutoencoderAD
        from dashboard.features import autoencoder as view
        with patch.object(view, 'DATA_PATH', self.root / 'missing'):
            feature = AutoencoderAD()
        artifact = feature._load_model(self.models / 'autoencoder.json')
        with patch.object(view, 'load_table', return_value=probe):
            figure, _ = feature._analyze('test', 'probe', artifact)
        score = next(trace for trace in figure.data if trace.name == "Reconstruction error")
        np.testing.assert_allclose(score.y, expected_errors, atol=1e-12)

    def test_diagnosis_training_writes_json_models(self):
        measurements = pd.DataFrame({'CC_1.speed': [0., 1., 2.]})
        training = self.root / 'data/training/0_0pct_test'
        measurements.to_parquet(training / 'measurements.parquet')
        pd.DataFrame({'CC_1.MotorFault': [False, True, False]}).to_parquet(training / 'faults.parquet')
        with patch.object(case_based, 'REPO_ROOT', self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            case_based.train()
        model = json.loads((self.models / 'case_based.json').read_text())
        result = case_based.predict(measurements)
        self.assertEqual(result['component'].tolist(), ['CC_1'] * 3)
        pd.DataFrame({'CC_1.MotorFault': [False, True]}).to_parquet(
            self.root / 'data/training/0_0pct_test/faults.parquet')
        with patch.object(structural_knowledge, 'REPO_ROOT', self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            structural_knowledge.train()
        knowledge = json.loads((self.models / 'structural_knowledge.json').read_text())
        self.assertTrue(knowledge['component_type_by_id'])

    def test_missing_models_warn_and_skip_without_loading_data(self):
        output = io.StringIO()
        with contextlib.nullcontext(), \
                patch.object(evaluate.pd, 'read_parquet') as load, \
                contextlib.redirect_stdout(output):
            self.assertEqual(evaluate.evaluate_anomaly_detection({}), {})
            self.assertEqual(evaluate.evaluate_diagnosis({}), {})
        load.assert_not_called()
        for filename in ('range_monitoring.json', 'autoencoder.json',
                         'case_based.json', 'structural_knowledge.json'):
            self.assertIn(filename, output.getvalue())
        self.assertEqual(output.getvalue().count('WARNING:'), 4)

    def test_missing_structural_thresholds_warn_and_skip(self):
        (self.models / 'structural_knowledge.json').write_text('{}')
        output = io.StringIO()
        with contextlib.nullcontext(), contextlib.redirect_stdout(output):
            self.assertEqual(evaluate.evaluate_diagnosis({}), {})
        self.assertIn('range_monitoring.json', output.getvalue())

    def test_available_detector_runs_when_other_model_is_missing(self):
        (self.models / 'range_monitoring.json').write_text(json.dumps({
            'speed': {'type': 'numeric', 'lower': 0., 'upper': 1.}}))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = evaluate.evaluate_anomaly_detection({'sample': {
                'measurements': pd.DataFrame({'speed': [0., 2.]}),
                'faults': pd.DataFrame({'fault': [False, True]}),
            }})
        self.assertEqual(list(result), ['Range Monitoring'])
        self.assertEqual(result['Range Monitoring']['F1'], 1.)
        self.assertIn('autoencoder.json', output.getvalue())

    def test_new_methods_are_discovered_without_registration(self):
        methods = self.root / 'methods'
        loaded = {}
        for task in ('ad', 'dx'):
            folder = methods / task
            folder.mkdir(parents=True)
            script = folder / 'new_method.py'
            source = "import pandas as pd\nMETHOD_NAME = 'New Method'\n"
            source += "def predict(measurements):\n"
            source += ("    return measurements['speed'].to_numpy() > 1\n" if task == 'ad' else
                       "    return pd.DataFrame({'component': 'CC_1', 'fault_type': 'MotorFault'}, index=measurements.index)\n")
            script.write_text(source)
            name = f'methods.{task}.new_method'
            spec = importlib.util.spec_from_file_location(name, script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = module
        data = {'sample': {'measurements': pd.DataFrame({'speed': [0., 2.]}),
                           'faults': pd.DataFrame({'CC_1.MotorFault': [False, True]})}}
        with patch.object(evaluate, 'REPO_ROOT', self.root), patch.dict(sys.modules, loaded), \
                contextlib.redirect_stdout(io.StringIO()):
            ad = evaluate.evaluate_anomaly_detection(data)
            dx = evaluate.evaluate_diagnosis(data)
        self.assertEqual(ad['New Method']['F1'], 1.)
        self.assertEqual(dx['New Method']['Fault_F1'], 1.)

    def test_existing_results_skip_prediction_for_both_tasks(self):
        with patch.object(range_monitoring, 'predict') as rm, \
                patch.object(autoencoder, 'predict') as ae, \
                patch.object(case_based, 'predict') as cb, \
                patch.object(structural_knowledge, 'predict') as sk, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.evaluate_anomaly_detection({}, {
                'Range Monitoring': {'F1': 0.5}, 'Vanilla Autoencoder': {'F1': 0.6}}), {})
            self.assertEqual(evaluate.evaluate_diagnosis({}, {
                'Case-Based': {'Fault_F1': 0.5}, 'Structural-Knowledge-Based': {'Fault_F1': 0.6}}), {})
        for predict in (rm, ae, cb, sk):
            predict.assert_not_called()

    def test_main_adds_results_and_leaves_existing_values_unchanged(self):
        (self.root / 'data/test').mkdir()
        ad_path = self.root / 'benchmark_ad.json'
        dx_path = self.root / 'benchmark_dx.json'
        ad_path.write_text('{"Old AD": {"F1": 0.25}}')
        dx_path.write_text('{"Old DX": {"Fault_F1": 0.5}}')
        with patch.object(evaluate, 'REPO_ROOT', self.root), \
                patch.object(evaluate, 'evaluate_anomaly_detection', return_value={'New AD': {'F1': 1.}}) as ad, \
                patch.object(evaluate, 'evaluate_diagnosis', return_value={'New DX': {'Fault_F1': 1.}}) as dx, \
                contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        self.assertEqual(json.loads(ad_path.read_text()), {'Old AD': {'F1': 0.25}, 'New AD': {'F1': 1.}})
        self.assertEqual(json.loads(dx_path.read_text()), {'Old DX': {'Fault_F1': 0.5}, 'New DX': {'Fault_F1': 1.}})
        before = (ad_path.read_bytes(), dx_path.read_bytes())
        with patch.object(evaluate, 'REPO_ROOT', self.root), \
                patch.object(evaluate, 'evaluate_anomaly_detection', return_value={}), \
                patch.object(evaluate, 'evaluate_diagnosis', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        self.assertEqual((ad_path.read_bytes(), dx_path.read_bytes()), before)

    def test_encoding_preserves_feature_order_and_handles_unseen_categories(self):
        model = {'columns': ['number', 'enabled', 'state'], 'encoding': {
            'number': {'type': 'numeric', 'fill_value': 2.},
            'enabled': {'type': 'boolean'},
            'state': {'type': 'categorical', 'categories': ['idle', 'run']},
        }}
        frame = pd.DataFrame({
            'state': ['run', 'unseen', None],
            'enabled': pd.Series([True, False, None], dtype='boolean'),
            'number': [1., np.nan, 3.],
        })
        expected = [[1., 1., 0., 1.], [2., 0., 0., 0.], [3., 0., 0., 0.]]
        np.testing.assert_array_equal(autoencoder.encode_measurements(frame, model), expected)
        from dashboard.features import autoencoder as view
        np.testing.assert_array_equal(view.encode_measurements(frame, model), expected)

    def test_structural_training_uses_only_observed_training_faults(self):
        # Remove the placeholder scenario: structural training requires faults.parquet.
        (self.root / 'data/training/0_0pct_test/faults.parquet').parent.rename(
            self.root / 'data/training/0_0pct_faults_rep_1')
        training = self.root / 'data/training/0_0pct_faults_rep_1'
        pd.DataFrame({
            'CC_1.MotorFault': [False, True],
            'CC_1.IncreasedDampingFault': [0., 0.1],
            'CC_2.NumericFault': [0., 0.5],
            'CC_2.InactiveFault': [0., 0.],
        }).to_parquet(training / 'faults.parquet')
        test = self.root / 'data/test/example'
        test.mkdir(parents=True)
        pd.DataFrame({'CC_1.TestOnlyFault': [True]}).to_parquet(test / 'faults.parquet')
        hierarchy = self.root / 'hierarchy.json'
        hierarchy.write_text(json.dumps({'areas': {'A': {'CC': ['CC_1', 'CC_2']}}}))
        with patch.object(structural_knowledge, 'REPO_ROOT', self.root), \
                patch.object(structural_knowledge, 'STRUCTURAL_HIERARCHY_PATH', hierarchy), \
                contextlib.redirect_stdout(io.StringIO()):
            structural_knowledge.train()
        model = json.loads(structural_knowledge.MODEL_PATH.read_text())
        self.assertEqual(model['faults_by_component'], {
            'CC_1': ['MotorFault'], 'CC_2': ['NumericFault']})
        self.assertEqual(model['faults_by_type'], {'CC': ['MotorFault', 'NumericFault']})
        structural_knowledge.THRESHOLDS_PATH.write_text(json.dumps({
            'CC_1.speed': {'type': 'numeric', 'lower': 0., 'upper': 1.}}))
        result = structural_knowledge.predict(pd.DataFrame({'CC_1.speed': [2.]}))
        self.assertEqual(tuple(result.iloc[0]), ('CC_1', 'MotorFault'))

    def test_both_dx_methods_use_all_training_folders_and_numeric_faults(self):
        first = self.root / 'data/training/0_0pct_test'
        renamed = first.with_name('0_0pct_faults_rep_1')
        first.rename(renamed)
        second = self.root / 'data/training/10_0pct_faults_rep_1'
        second.mkdir()
        for folder, state, fault in ((renamed, 'blocked', 'MotorFault'), (second, 'jammed', 'NumericFault')):
            pd.DataFrame({'CC_1.speed': [0., 1., 0.], 'CC_1.enabled': [False, True, False],
                          'CC_1.state': ['idle', state, 'idle']}).to_parquet(folder / 'measurements.parquet')
            pd.DataFrame({f'CC_1.{fault}': [0., 0.5, 0.],
                          'CC_1.IncreasedDampingFault': [0.1] * 3}).to_parquet(folder / 'faults.parquet')
        test = self.root / 'data/test/example'
        test.mkdir(parents=True)
        pd.DataFrame({'CC_1.TestOnlyFault': [True]}).to_parquet(test / 'faults.parquet')
        hierarchy = self.root / 'hierarchy.json'
        hierarchy.write_text(json.dumps({'areas': {'A': {'CC': ['CC_1']}}}))
        with patch.object(case_based, 'REPO_ROOT', self.root), \
                patch.object(structural_knowledge, 'REPO_ROOT', self.root), \
                patch.object(structural_knowledge, 'STRUCTURAL_HIERARCHY_PATH', hierarchy), \
                contextlib.redirect_stdout(io.StringIO()):
            case_based.train()
            structural_knowledge.train()
        model = json.loads(case_based.MODEL_PATH.read_text())
        self.assertEqual({case['fault_label'] for case in model['cases']},
                         {'CC_1.MotorFault', 'CC_1.NumericFault'})
        self.assertEqual(model['encoding']['CC_1.enabled']['type'], 'boolean')
        self.assertEqual(model['encoding']['CC_1.state']['categories'], ['blocked', 'idle', 'jammed'])
        probe = pd.DataFrame({'CC_1.speed': [1., 1.], 'CC_1.enabled': [True, True],
                              'CC_1.state': ['blocked', 'jammed']})
        self.assertEqual(case_based.predict(probe)['fault_type'].tolist(), ['MotorFault', 'NumericFault'])
        from dashboard.features import case_based as view
        np.testing.assert_array_equal(view.encode_measurements(probe, model),
                                      case_based.encode_measurements(probe, model))
        knowledge = json.loads(structural_knowledge.MODEL_PATH.read_text())
        self.assertEqual(knowledge['faults_by_component'], {'CC_1': ['MotorFault', 'NumericFault']})

    def test_events_do_not_merge_across_scenario_boundaries(self):
        self.assertAlmostEqual(evaluate._composite_f1(
            np.array([True, True]), np.array([True, False]),
            [np.array([True]), np.array([True])]), 2 / 3)

    def test_misaligned_labels_are_rejected(self):
        (self.models / 'range_monitoring.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'aligned'):
            evaluate.evaluate_anomaly_detection({'example': {
                'measurements': pd.DataFrame({'speed': [0.]}, index=[1]),
                'faults': pd.DataFrame({'fault': [False]}, index=[2]),
            }})

    def test_missing_model_inputs_are_rejected(self):
        range_monitoring.MODEL_PATH.write_text(json.dumps({'speed': {}}))
        autoencoder.MODEL_PATH.write_text(json.dumps({'columns': ['speed']}))
        with self.assertRaisesRegex(ValueError, 'Missing measurement'):
            range_monitoring.predict(pd.DataFrame({'other': [0]}))
        with self.assertRaisesRegex(ValueError, 'Missing measurement'):
            autoencoder.predict(pd.DataFrame({'other': [0]}))

    def test_structural_diagnosis_does_not_use_true_fault_type(self):
        model = {'component_type_by_id': {'CC_1': 'CC'},
                 'faults_by_type': {'CC': ['fault_a', 'fault_b']}}
        thresholds = {'CC_1.speed': {'type': 'numeric', 'lower': 0., 'upper': 1.}}
        structural_knowledge.MODEL_PATH.write_text(json.dumps(model))
        structural_knowledge.THRESHOLDS_PATH.write_text(json.dumps(thresholds))
        result = structural_knowledge.predict(pd.DataFrame({"CC_1.speed": [2.]}))
        self.assertEqual(tuple(result.iloc[0]), ("CC_1", "unknown"))


if __name__ == '__main__':
    unittest.main()
