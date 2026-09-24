"""Train numeric/categorical range thresholds and save the model for evaluation."""

import json
from pathlib import Path

import pandas as pd



METHOD_NAME = "Range Monitoring"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PERCENTILE_MARGIN = 0.0
MODEL_PATH = REPO_ROOT / "models" / "range_monitoring.json"


def predict_column_anomalies(
    data: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Return per-column violations of the saved numeric/categorical thresholds."""

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

        if config.get("type", "numeric") == "numeric":
            mask = (
                series.isna()
                | (series < config.get("lower", config.get("min")))
                | (series > config.get("upper", config.get("max")))
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

    return anomaly_mask


def predict(measurements: pd.DataFrame):
    """Return True for rows containing at least one anomaly."""
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return predict_column_anomalies(measurements, model).any(axis=1).to_numpy(dtype=bool)


def train() -> None:
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

    output_path = MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    train()
