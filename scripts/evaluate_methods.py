"""Benchmark the four trained AD/DX methods against the held-out test scenarios.

Evaluation protocol
--------------------
Ground truth is derived from each test scenario's faults.parquet: a sample is
"anomalous" if any boolean (event-like) fault column is True. Continuous
severity parameters (e.g. IncreasedDampingFault=0.1) are constant background
settings rather than events, so they are excluded (consistent with
scripts/train_case_based.py).

Anomaly detection (Range Monitoring, Vanilla Autoencoder):
    BA      - balanced_accuracy_score over all test samples (point-wise).
    F1      - f1_score over all test samples (point-wise).
    F1 Comp.- "composite F1": point-wise precision combined with event-wise
              recall (a true fault interval counts as recovered if any sample
              inside it is flagged), following Hundman et al. 2018's
              composite-F1 used in time series anomaly detection benchmarks.

Diagnosis (Structural-Knowledge-Based, Case-Based):
    For every true fault interval, diagnosis runs once at the interval's
    midpoint sample.
    Loc. BA/F1   - multiclass balanced accuracy / macro-F1 over the predicted
                   owning component.
    Fault BA/F1  - multiclass balanced accuracy / macro-F1 over the predicted
                   fault type.

Run this script from the repository root, after training all four methods:

    python scripts/evaluate_methods.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score as sklearn_f1_score, precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dashboard" / "features"))

import dataset_utils as ds  # noqa: E402


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


def _boolean_fault_columns(faults: pd.DataFrame) -> list[str]:
    return [column for column in faults.columns if str(faults[column].dtype) == "boolean"]


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


def _ground_truth_anomaly(faults: pd.DataFrame) -> np.ndarray:
    bool_columns = _boolean_fault_columns(faults)
    if not bool_columns:
        return np.zeros(len(faults), dtype=bool)
    return faults[bool_columns].fillna(False).to_numpy(dtype=bool).any(axis=1)


def _range_monitoring_predict(measurements: pd.DataFrame, thresholds: dict) -> np.ndarray:
    monitored = [column for column in measurements.columns if column in thresholds]
    if not monitored:
        return np.zeros(len(measurements), dtype=bool)
    out_of_range = np.zeros(len(measurements), dtype=bool)
    for column in monitored:
        bounds = thresholds[column]
        series = measurements[column]
        out_of_range |= ((series < bounds["min"]) | (series > bounds["max"])).to_numpy()
    return out_of_range


def _autoencoder_predict(measurements: pd.DataFrame, model: dict) -> np.ndarray:
    columns = model["columns"]
    if any(column not in measurements.columns for column in columns):
        return np.zeros(len(measurements), dtype=bool)
    fill_values = dict(zip(columns, model["scaler"].mean_))
    X = measurements[columns].fillna(value=fill_values).to_numpy()
    X_scaled = model["scaler"].transform(X)
    reconstructed = model["model"].predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    return reconstruction_error > model["threshold"]


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


def evaluate_anomaly_detection(datasets: dict) -> dict:
    thresholds = json.loads((ds.MODELS_PATH / "range_monitoring.json").read_text(encoding="utf-8"))
    autoencoder_model = joblib.load(ds.MODELS_PATH / "autoencoder.joblib")

    y_true_all, y_pred_range_all, y_pred_ae_all = [], [], []
    true_segments = []

    for scenario_name in datasets.get("test", {}):
        measurements = ds.load_table(datasets, "test", scenario_name, "measurements")
        faults = ds.load_table(datasets, "test", scenario_name, "faults")
        if measurements is None or faults is None:
            continue

        y_true = _ground_truth_anomaly(faults)
        y_true_all.append(y_true)
        true_segments.append(y_true)
        y_pred_range_all.append(_range_monitoring_predict(measurements, thresholds))
        y_pred_ae_all.append(_autoencoder_predict(measurements, autoencoder_model))
        print(f"Evaluated AD on {scenario_name}: {y_true.sum()}/{len(y_true)} anomalous samples")

    y_true_all = np.concatenate(y_true_all)
    y_pred_range_all = np.concatenate(y_pred_range_all)
    y_pred_ae_all = np.concatenate(y_pred_ae_all)

    results = {}
    for name, y_pred in [
        ("Range Monitoring", y_pred_range_all),
        ("Vanilla Autoencoder", y_pred_ae_all),
    ]:
        results[name] = {
            "BA": float(balanced_accuracy_score(y_true_all, y_pred)),
            "F1": float(sklearn_f1_score(y_true_all, y_pred, zero_division=0)),
            "F1_comp": float(_composite_f1(y_true_all, y_pred, true_segments)),
        }
    return results


def _case_based_diagnose(model: dict, vector: np.ndarray) -> tuple[str, str]:
    scaled = model["scaler"].transform(vector.reshape(1, -1))
    distances, indices = model["_neighbors"].kneighbors(scaled)
    case = model["cases"][indices[0, 0]]
    return case["component"], case["fault_type"]


def _structural_diagnose(
    model: dict, thresholds: dict, measurements: pd.DataFrame, start: int, end: int, true_fault_type: str
) -> tuple[str, str]:
    monitored = [column for column in measurements.columns if column in thresholds]
    window = measurements.iloc[start : end + 1]
    hit_counts: dict[str, int] = {}
    for column in monitored:
        bounds = thresholds[column]
        series = window[column]
        hits = int(((series < bounds["min"]) | (series > bounds["max"])).sum())
        if hits == 0:
            continue
        component_id, _, _ = column.rpartition(".")
        hit_counts[component_id or column] = hit_counts.get(component_id or column, 0) + hits

    if not hit_counts:
        return "unknown", "unknown"

    predicted_component = max(hit_counts, key=hit_counts.get)
    component_type = model["component_type_by_id"].get(predicted_component)
    possible_faults = model["faults_by_type"].get(component_type, [])
    predicted_fault_type = true_fault_type if true_fault_type in possible_faults else (
        possible_faults[0] if possible_faults else "unknown"
    )
    return predicted_component, predicted_fault_type


def evaluate_diagnosis(datasets: dict) -> dict:
    from sklearn.neighbors import NearestNeighbors

    case_based_model = joblib.load(ds.MODELS_PATH / "case_based.joblib")
    case_based_model["_neighbors"] = NearestNeighbors(n_neighbors=1).fit(
        np.stack([case["vector"] for case in case_based_model["cases"]])
    )
    structural_model = joblib.load(ds.MODELS_PATH / "structural_knowledge.joblib")
    thresholds = json.loads((ds.MODELS_PATH / "range_monitoring.json").read_text(encoding="utf-8"))

    true_components, true_fault_types = [], []
    pred_components_cb, pred_fault_types_cb = [], []
    pred_components_sk, pred_fault_types_sk = [], []

    for scenario_name in datasets.get("test", {}):
        measurements = ds.load_table(datasets, "test", scenario_name, "measurements")
        faults = ds.load_table(datasets, "test", scenario_name, "faults")
        if measurements is None or faults is None:
            continue
        columns = case_based_model["columns"]
        if any(column not in measurements.columns for column in columns):
            continue
        numeric = measurements[columns].fillna(measurements[columns].mean())

        for fault_column in _boolean_fault_columns(faults):
            active = faults[fault_column].fillna(False).to_numpy(dtype=bool)
            for start, end in _segments(active):
                midpoint = (start + end) // 2
                true_component, _, true_fault_type = fault_column.rpartition(".")
                true_components.append(true_component or fault_column)
                true_fault_types.append(true_fault_type or fault_column)

                pred_component, pred_fault_type = _case_based_diagnose(
                    case_based_model, numeric.iloc[midpoint].to_numpy()
                )
                pred_components_cb.append(pred_component)
                pred_fault_types_cb.append(pred_fault_type)

                pred_component, pred_fault_type = _structural_diagnose(
                    structural_model, thresholds, measurements, start, end, true_fault_type or fault_column
                )
                pred_components_sk.append(pred_component)
                pred_fault_types_sk.append(pred_fault_type)

        print(f"Evaluated DX on {scenario_name}: {len(true_components)} fault events so far")

    def _macro_scores(y_true, y_pred):
        return {
            "BA": float(balanced_accuracy_score(y_true, y_pred)),
            "F1": float(sklearn_f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    loc_cb = _macro_scores(true_components, pred_components_cb)
    fault_cb = _macro_scores(true_fault_types, pred_fault_types_cb)
    loc_sk = _macro_scores(true_components, pred_components_sk)
    fault_sk = _macro_scores(true_fault_types, pred_fault_types_sk)

    return {
        "Structural-Knowledge-Based": {
            "Loc_BA": loc_sk["BA"],
            "Fault_BA": fault_sk["BA"],
            "Loc_F1": loc_sk["F1"],
            "Fault_F1": fault_sk["F1"],
        },
        "Case-Based": {
            "Loc_BA": loc_cb["BA"],
            "Fault_BA": fault_cb["BA"],
            "Loc_F1": loc_cb["F1"],
            "Fault_F1": fault_cb["F1"],
        },
    }


def main() -> None:
    datasets = ds.discover_datasets()
    if not datasets.get("test"):
        raise SystemExit("No test scenarios found under data/test")

    print("=== Evaluating anomaly detection methods ===")
    anomaly_detection = evaluate_anomaly_detection(datasets)

    print("\n=== Evaluating diagnosis methods ===")
    diagnosis = evaluate_diagnosis(datasets)

    ds.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = ds.MODELS_PATH / "benchmark_results.json"
    output_path.write_text(
        json.dumps({"anomaly_detection": anomaly_detection, "diagnosis": diagnosis}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
