"""
Anomaly detection evaluation metrics.

This module implements metrics for evaluating anomaly detection methods:
- Balanced accuracy (BA)
- F1-score (F1)
- Composite F1-score (Fc1) with event-wise recall
"""

import numpy as np


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
