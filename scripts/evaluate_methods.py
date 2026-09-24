"""Benchmark all discovered AD/DX methods against the held-out test scenarios.

Evaluation protocol
--------------------
Ground truth is derived from each test scenario's faults.parquet: a sample is
"anomalous" if any fault column is nonzero. For the current training/test
recordings, IncreasedDampingFault values of both 0 and 0.1 are normal.
Missing values are treated as inactive.

Anomaly detection (Range Monitoring, Vanilla Autoencoder):
    BA      - balanced_accuracy_score over all test samples (point-wise).
    F1      - f1_score over all test samples (point-wise).
    F1 Comp.- "composite F1": point-wise precision combined with event-wise
              recall (a true fault interval counts as recovered if any sample
              inside it is flagged), following Hundman et al. 2018's
              composite-F1 used in time series anomaly detection benchmarks.
    Confusion counts, precision, recall, accuracy, and error rates are also
    reported overall and per scenario for both detectors.

Diagnosis (Structural-Knowledge-Based, Case-Based):
    For every true fault interval, diagnosis runs once at the interval's
    last faulty sample (before the fault clears).
    Loc. BA/F1   - multiclass balanced accuracy / macro-F1 over the predicted
                   owning component.
    Fault BA/F1  - multiclass balanced accuracy / macro-F1 over the predicted
                   fault type.

Run this script from the repository root, after training the methods to evaluate:

    python scripts/evaluate_methods.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score as sklearn_f1_score, precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

METHODS_PATH = REPO_ROOT / "methods"
DATA_PATH = Path(__file__).resolve().parents[1] / "data"


FILENAMES_WITH_INCREASED_DUMPING_CONFLICT = [
    "0_0pct_faults_rep_1",
    "0_0pct_faults_rep_2",
    "0_0pct_faults_rep_3",
    "10_0pct_faults_rep_1",
    "10_0pct_faults_rep_2",
    "10_0pct_faults_rep_3",
    "1_0pct_faults_rep_1",
    "1_0pct_faults_rep_2",
    "1_0pct_faults_rep_3",
    "AMR_30pct_faults",
    "AMR_multi_2_30pct_faults",
    "PAR_30pct_faults",
    "PAR_multi_2_30pct_faults",
    "VZ_30pct_faults",
    "VZ_multi_2_30pct_faults",
    "WA_30pct_faults",
    "WA_multi_2_30pct_faults",
    "WE_30pct_faults",
    "WE_multi_2_30pct_faults",
]


def balanced_accuracy(tp, tn, fp, fn):
    """
    Calculate point-wise anomaly detection balanced accuracy.

    BA = 1/2 * (TP/(TP+FN) + TN/(TN+FP))

    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives

    Returns:
        Balanced accuracy score
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (sensitivity + specificity)


def precision(tp, fp):
    """
    Calculate point-wise precision.

    P = TP / (TP + FP)

    Args:
        tp: True positives
        fp: False positives

    Returns:
        Precision score
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    """
    Calculate point-wise recall.

    R = TP / (TP + FN)

    Args:
        tp: True positives
        fn: False negatives

    Returns:
        Recall score
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(tp, fp, fn):
    """
    Calculate point-wise anomaly detection F1-score.

    F1 = 2 * P * R / (P + R)

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        F1-score
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def event_wise_recall(tp_event, fn_event):
    """
    Calculate event-wise recall.

    R_event = TP_event / (TP_event + FN_event)

    Args:
        tp_event: Number of detected anomalous segments
        fn_event: Number of missed anomalous segments

    Returns:
        Event-wise recall score
    """
    return tp_event / (tp_event + fn_event) if (tp_event + fn_event) > 0 else 0.0


def composite_f1_score(tp, fp, tp_event, fn_event):
    """
    Calculate composite F1-score for time-series anomaly detection.

    Fc1 = 2 * P * R_event / (P + R_event)

    Args:
        tp: Point-wise true positives
        fp: Point-wise false positives
        tp_event: Number of detected anomalous segments
        fn_event: Number of missed anomalous segments

    Returns:
        Composite F1-score
    """
    p = precision(tp, fp)
    r_event = event_wise_recall(tp_event, fn_event)
    return (2 * p * r_event) / (p + r_event) if (p + r_event) > 0 else 0.0


def compute_point_wise_metrics(y_true, y_pred):
    """
    Compute point-wise TP, TN, FP, FN from ground truth and predictions.

    Args:
        y_true: Ground truth binary labels (numpy array)
        y_pred: Predicted binary labels (numpy array)

    Returns:
        Tuple of (tp, tn, fp, fn)
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    return int(tp), int(tn), int(fp), int(fn)


def compute_event_wise_metrics(y_true, y_pred):
    """
    Compute event-wise TP_event and FN_event from ground truth and predictions.

    An event is a contiguous segment of anomalous points in y_true.
    TP_event counts segments where at least one point is correctly detected.
    FN_event counts segments that are completely missed.

    Args:
        y_true: Ground truth binary labels (numpy array)
        y_pred: Predicted binary labels (numpy array)

    Returns:
        Tuple of (tp_event, fn_event)
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    # Find anomalous segments in ground truth
    segments = []
    in_segment = False
    start = 0

    for i in range(len(y_true)):
        if y_true[i] and not in_segment:
            start = i
            in_segment = True
        elif not y_true[i] and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(y_true)))

    # Count detected and missed segments
    tp_event = 0
    fn_event = 0

    for start, end in segments:
        if np.any(y_pred[start:end]):
            tp_event += 1
        else:
            fn_event += 1

    return tp_event, fn_event


def _segments(active: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) integer-position pairs of contiguous True runs."""
    segments = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(active) - 1))
    return segments


def _composite_f1(y_true: np.ndarray, y_pred: np.ndarray, segment_lengths: list[np.ndarray]) -> float:
    precision = precision_score(y_true, y_pred, zero_division=0)
    total_segments = sum(len(_segments(segment)) for segment in segment_lengths)
    if total_segments == 0 or precision == 0:
        return 0.0
    detected = 0
    offset = 0
    for i, segment in enumerate(segment_lengths):
        for start, end in _segments(segment):
            if y_pred[offset + start : offset + end + 1].any():
                detected += 1
        offset += len(segment)
    recall_event = detected / total_segments
    if precision + recall_event == 0:
        return 0.0
    return 2 * precision * recall_event / (precision + recall_event)


def calculate_metrics(y_true, y_pred) -> dict:
    """Use the same point-wise scoring for every anomaly detector."""
    tp, tn, fp, fn = compute_point_wise_metrics(y_true, y_pred)
    divide = lambda n, d: n / d if d else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision(tp, fp), "recall": recall(tp, fn),
        "specificity": divide(tn, tn + fp),
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "f1": f1_score(tp, fp, fn),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
        "BA": balanced_accuracy(tp, tn, fp, fn),
        "F1": f1_score(tp, fp, fn),
    }

def extract_fault_labels(faults: pd.DataFrame, filename: str | None = None) -> pd.DataFrame:
    """Mark nonzero fault values, excluding the current datasets' damping baseline."""

    labels = faults.fillna(0).ne(0)
    # In the current training/test recordings, 0.1 is also normal for this fault.
    for column in faults.columns:
        if filename and filename in FILENAMES_WITH_INCREASED_DUMPING_CONFLICT and column.rsplit(".", 1)[-1] == "IncreasedDampingFault":
            labels[column] &= faults[column].ne(0.1).fillna(False)
    return labels


def diagnosis_events(labels: pd.DataFrame) -> list[tuple[int, str]]:
    """Return the last active row and fault label for each per-fault segment."""
    return sorted((end, column) for column in labels
                  for _, end in _segments(labels[column].to_numpy(dtype=bool)))


def evaluate_anomaly_detection(datasets: dict, existing_results: dict | None = None) -> dict:
    modules = [importlib.import_module(f"methods.ad.{p.stem}") for p in (REPO_ROOT / "methods/ad").glob("[!_]*.py")]
    models = {}
    for module in modules:
        if not callable(getattr(module, "predict", None)):
            continue
        name = getattr(module, "METHOD_NAME", module.__name__)
        if name in (existing_results or {}):
            print(f"Skipping {name}: benchmark result already exists")
            continue
        paths = [getattr(module, key) for key in ("MODEL_PATH", "THRESHOLDS_PATH") if hasattr(module, key)]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            print(f"WARNING: Skipping {name}: model JSON not found: {missing[0]}")
            continue
        models[name] = module.predict
    if not models:
        return {}

    true_segments: list[np.ndarray] = []
    predictions = {name: [] for name in models}
    scenarios = {name: {} for name in models}

    for scenario_name in datasets.keys():
        measurements = datasets[scenario_name]["measurements"]
        faults = extract_fault_labels(datasets[scenario_name]["faults"], scenario_name)

        if measurements.empty or not measurements.index.equals(faults.index):
            raise ValueError(f"{scenario_name}: nonempty measurements and aligned fault indexes are required")
        y_true = faults.fillna(False).to_numpy(dtype=bool).any(axis=1)

        true_segments.append(y_true)
        for name, predict in models.items():
            y_pred = predict(measurements)
            predictions[name].append(y_pred)
            scenarios[name][scenario_name] = {
                **calculate_metrics(y_true, y_pred),
                "F1_comp": float(_composite_f1(y_true, y_pred, [y_true])),
            }
        print(f"Evaluated AD on {scenario_name}: {y_true.sum()}/{len(y_true)} anomalous samples")
    if not true_segments:
        raise ValueError("No test scenarios available for evaluation")
    y_true = np.concatenate(true_segments)
    results = {}
    for name in models:
        y_pred = np.concatenate(predictions[name])
        results[name] = {
            **calculate_metrics(y_true, y_pred),
            "F1_comp": float(_composite_f1(y_true, y_pred, true_segments)),
            "scenarios": scenarios[name],
        }
    return results


def evaluate_diagnosis(datasets: dict, existing_results: dict | None = None) -> dict:
    modules = [importlib.import_module(f"methods.dx.{p.stem}") for p in (REPO_ROOT / "methods/dx").glob("[!_]*.py")]
    models = {}
    for module in modules:
        if not callable(getattr(module, "predict", None)):
            continue
        name = getattr(module, "METHOD_NAME", module.__name__)
        if name in (existing_results or {}):
            print(f"Skipping {name}: benchmark result already exists")
            continue
        paths = [getattr(module, key) for key in ("MODEL_PATH", "THRESHOLDS_PATH") if hasattr(module, key)]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            print(f"WARNING: Skipping {name}: model JSON not found: {missing[0]}")
            continue
        models[name] = module.predict
    if not models:
        return {}

    true_components, true_fault_types = [], []
    predictions = {name: {"component": [], "fault_type": []} for name in models}
    for scenario_name in datasets:
        measurements = datasets[scenario_name]["measurements"]
        faults = extract_fault_labels(datasets[scenario_name]["faults"], scenario_name)
        if measurements.empty or not measurements.index.equals(faults.index):
            raise ValueError(f"{scenario_name}: nonempty measurements and aligned fault indexes are required")
        events = diagnosis_events(faults)
        if not events:
            continue
        for _, fault_column in events:
            component, _, fault_type = fault_column.rpartition(".")
            true_components.append(component or fault_column)
            true_fault_types.append(fault_type or fault_column)
        samples = measurements.iloc[[position for position, _ in events]]
        for name, predict in models.items():
            result = predict(samples)
            for column in ("component", "fault_type"):
                predictions[name][column].extend(result[column].tolist())
        print(f"Evaluated DX on {scenario_name}: {len(true_components)} fault events so far")

    if not true_components:
        raise ValueError("No fault events available for diagnosis evaluation")
    results = {}
    for name, predicted in predictions.items():
        results[name] = {}
        for prefix, target, column in (
            ("Loc", true_components, "component"),
            ("Fault", true_fault_types, "fault_type"),
        ):
            results[name][f"{prefix}_BA"] = float(balanced_accuracy_score(target, predicted[column]))
            results[name][f"{prefix}_F1"] = float(
                sklearn_f1_score(target, predicted[column], average="macro", zero_division=0)
            )
    return results


def main() -> None:
    test_dir = REPO_ROOT / "data" / "test"

    test_data = {
        p.name: {"measurements": pd.read_parquet(p / "measurements.parquet"),
                 "faults": pd.read_parquet(p / "faults.parquet"),
                 "parameters": pd.read_parquet(p / "parameters.parquet")
                 }
        for p in sorted(test_dir.iterdir()) if p.is_dir()
    }

    print("=== Evaluating anomaly detection methods ===")
    ad_output_path = REPO_ROOT / "benchmark_ad.json"
    ad_results = json.loads(ad_output_path.read_text(encoding="utf-8")) if ad_output_path.exists() else {}
    if not isinstance(ad_results, dict):
        raise ValueError(f"{ad_output_path} must contain a JSON object")
    new_ad_results = evaluate_anomaly_detection(test_data, ad_results)
    if new_ad_results:
        ad_results.update(new_ad_results)
        ad_output_path.write_text(json.dumps(ad_results, indent=2), encoding="utf-8")
        print(f"Saved new anomaly detection benchmark results to {ad_output_path}")

    print("=== Evaluating diagnosis methods ===")
    dx_output_path = REPO_ROOT / "benchmark_dx.json"
    dx_results = json.loads(dx_output_path.read_text(encoding="utf-8")) if dx_output_path.exists() else {}
    if not isinstance(dx_results, dict):
        raise ValueError(f"{dx_output_path} must contain a JSON object")
    new_dx_results = evaluate_diagnosis(test_data, dx_results)
    if new_dx_results:
        dx_results.update(new_dx_results)
        dx_output_path.write_text(json.dumps(dx_results, indent=2), encoding="utf-8")
        print(f"Saved new diagnosis benchmark results to {dx_output_path}")


if __name__ == "__main__":
    main()
