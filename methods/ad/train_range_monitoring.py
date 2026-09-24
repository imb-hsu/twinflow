"""Train and evaluate the range-monitoring anomaly detector.

Estimates per-column thresholds from all training recordings and evaluates
them against fault labels in the test recordings.
"""

from pathlib import Path

import mlflow
import pandas as pd


EXPERIMENT_NAME = "HSU TwinFlow"
mlflow.set_experiment(EXPERIMENT_NAME)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PERCENTILE_MARGIN = 0.0


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Calculate binary classification metrics."""

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def predict_anomalies(
    data: pd.DataFrame,
    thresholds: dict,
) -> pd.Series:
    """Return True for rows containing at least one anomaly."""

    missing_columns = set(thresholds) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing measurement columns: {sorted(missing_columns)}"
        )

    anomaly_mask = pd.DataFrame(
        False,
        index=data.index,
        columns=thresholds.keys(),
    )

    for column, config in thresholds.items():
        series = data[column]

        if config["type"] == "numeric":
            mask = (
                series.isna()
                | (series < config["lower"])
                | (series > config["upper"])
            )

        elif config["type"] == "categorical":
            mask = (
                series.isna()
                | ~series.astype(str).isin(config["categories"])
            )

        else:
            raise ValueError(
                f"Unknown threshold type '{config['type']}' "
                f"for column '{column}'"
            )

        anomaly_mask[column] = mask

    # Row is anomalous if ANY measurement is anomalous.
    return anomaly_mask.any(axis=1)


def get_fault_labels(faults: pd.DataFrame) -> pd.Series:
    """Return True if any fault column contains a non-zero value."""

    if faults.isna().any().any():
        raise ValueError("faults.parquet contains missing values.")

    # Assumes every column in faults.parquet represents a fault.
    return faults.ne(0).any(axis=1)


def evaluate(
    thresholds: dict,
) -> dict:
    test_dir = REPO_ROOT / "data" / "test"

    results = {}

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for dataset_dir in sorted(test_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        measurements_path = dataset_dir / "measurements.parquet"
        faults_path = dataset_dir / "faults.parquet"

        if not measurements_path.exists():
            raise FileNotFoundError(
                f"Missing {measurements_path}"
            )

        if not faults_path.exists():
            raise FileNotFoundError(
                f"Missing {faults_path}"
            )

        print(f"Evaluating '{dataset_dir.name}'")

        measurements = pd.read_parquet(measurements_path)
        faults = pd.read_parquet(faults_path)

        if len(measurements) != len(faults):
            raise ValueError(
                f"'{dataset_dir.name}' has different numbers of rows: "
                f"{len(measurements)} measurements vs "
                f"{len(faults)} fault labels."
            )

        # Reset indexes so comparison is strictly row-by-row.
        measurements = measurements.reset_index(drop=True)
        faults = faults.reset_index(drop=True)

        y_pred = predict_anomalies(
            measurements,
            thresholds,
        )

        y_true = get_fault_labels(faults)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        results[dataset_dir.name] = {
            "num_rows": len(measurements),
            "num_fault_rows": int(y_true.sum()),
            "num_predicted_anomaly_rows": int(y_pred.sum()),
            **metrics,
        }

        total_tp += metrics["tp"]
        total_tn += metrics["tn"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        print(
            f"  TP={metrics['tp']} "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']}"
        )

        print(
            f"  Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    # Calculate global metrics from the accumulated confusion matrix.
    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    specificity = safe_divide(total_tn, total_tn + total_fp)
    accuracy = safe_divide(
        total_tp + total_tn,
        total_tp + total_tn + total_fp + total_fn,
    )
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    overall = {
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "false_positive_rate": safe_divide(
            total_fp,
            total_fp + total_tn,
        ),
        "false_negative_rate": safe_divide(
            total_fn,
            total_fn + total_tp,
        ),
    }

    results["overall"] = overall

    return results


def main() -> None:
    training_dir = REPO_ROOT / "data" / "training"

    train_data = {
        p.name: pd.read_parquet(p / "measurements.parquet")
        for p in training_dir.iterdir()
        if p.is_dir()
    }

    train_data = pd.concat(
        train_data.values(),
        axis=0,
        ignore_index=True,
    )

    with mlflow.start_run(run_name="range_monitoring"):

        # --------------------
        # TRAINING
        # --------------------

        mlflow.log_param(
            "percentile_margin",
            PERCENTILE_MARGIN,
        )

        thresholds = {}

        for column in train_data.columns:
            print(f"Processing column '{column}'")

            if train_data[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values."
                )

            if (
                pd.api.types.is_numeric_dtype(train_data[column])
                and not pd.api.types.is_bool_dtype(train_data[column])
            ):
                lower = float(
                    train_data[column].quantile(
                        PERCENTILE_MARGIN
                    )
                )

                upper = float(
                    train_data[column].quantile(
                        1 - PERCENTILE_MARGIN
                    )
                )

                thresholds[column] = {
                    "type": "numeric",
                    "lower": lower,
                    "upper": upper,
                }

            else:
                categories = (
                    train_data[column]
                    .astype(str)
                    .unique()
                    .tolist()
                )

                thresholds[column] = {
                    "type": "categorical",
                    "categories": categories,
                }

        mlflow.log_dict(
            thresholds,
            "range_monitoring.json",
        )

        print(
            f"Logged thresholds for {len(thresholds)} "
            f"columns to MLflow"
        )

        # --------------------
        # EVALUATION
        # --------------------

        evaluation_results = evaluate(thresholds)

        overall = evaluation_results["overall"]

        mlflow.log_metrics({
            "test_tp": float(overall["tp"]),
            "test_tn": float(overall["tn"]),
            "test_fp": float(overall["fp"]),
            "test_fn": float(overall["fn"]),
            "test_precision": overall["precision"],
            "test_recall": overall["recall"],
            "test_specificity": overall["specificity"],
            "test_accuracy": overall["accuracy"],
            "test_f1": overall["f1"],
            "test_false_positive_rate": (
                overall["false_positive_rate"]
            ),
            "test_false_negative_rate": (
                overall["false_negative_rate"]
            ),
        })

        mlflow.log_dict(
            evaluation_results,
            "evaluation/results.json",
        )

        print("\nOverall evaluation")
        print("------------------")
        print(f"TP: {overall['tp']}")
        print(f"TN: {overall['tn']}")
        print(f"FP: {overall['fp']}")
        print(f"FN: {overall['fn']}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall:    {overall['recall']:.4f}")
        print(f"F1:        {overall['f1']:.4f}")
        print(f"Accuracy:  {overall['accuracy']:.4f}")


if __name__ == "__main__":
    main()