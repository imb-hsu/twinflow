"""Learn numeric/categorical thresholds separately for each component state."""

import json
import re
import sys
from pathlib import Path

import pandas as pd



METHOD_NAME = "Range Monitoring"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from methods.ad._signals import is_ad_signal, warn_missing_values

PERCENTILE_MARGIN = 0.0
MODEL_PATH = REPO_ROOT / "models" / "range_monitoring.json"


def state_column_for_signal(column: str, columns: set[str]) -> str | None:
    """Find the closest owning component's state, including nested parts and axes."""
    component, _, _ = str(column).rpartition(".")
    while component:
        state_column = f"{component}.CurrentState"
        if state_column in columns:
            return state_column
        component, _, _ = component.rpartition(".")
    return None


def violation_flags(series: pd.Series, config: dict, state_keys: pd.Series | None = None) -> pd.Series:
    """Score one signal against its state profile, or legacy global limits."""
    if "state_column" in config:
        if state_keys is None:
            raise ValueError(f"Missing component state column: {config['state_column']}")
        profiles = config["states"]
        # Unseen states are anomalous; missing states cannot select a profile.
        flags = series.notna() & state_keys.notna() & ~state_keys.isin(profiles)
        for state, profile in profiles.items():
            selected = state_keys.eq(state).fillna(False)
            flags |= selected & violation_flags(series, profile)
        return flags.fillna(False).astype(bool)

    if config.get("type", "numeric") == "numeric":
        mask = ((series < config.get("lower", config.get("min")))
                | (series > config.get("upper", config.get("max"))))
    elif config["type"] == "categorical":
        mask = ~series.astype(str).isin(config["categories"])
    else:
        raise ValueError(f"Unknown threshold type: {config['type']}")
    return (series.notna() & mask).fillna(False).astype(bool)


def predict_column_anomalies(
    data: pd.DataFrame,
    thresholds: dict,
    *,
    warn: bool = True,
) -> pd.DataFrame:
    """Return per-column violations using each component's current state."""

    thresholds = {column: config for column, config in thresholds.items() if is_ad_signal(column)}
    state_columns = {config["state_column"] for config in thresholds.values() if "state_column" in config}
    required_columns = set(thresholds) | state_columns
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing measurement columns: {sorted(missing_columns)}"
        )
    if warn:
        warn_missing_values(data, sorted(required_columns))
    state_keys = {column: data[column].astype("string") for column in state_columns}

    anomaly_mask = pd.DataFrame(
        False,
        index=data.index,
        columns=thresholds.keys(),
    )

    for column, config in thresholds.items():
        anomaly_mask[column] = violation_flags(data[column], config, state_keys.get(config.get("state_column")))

    return anomaly_mask


def predict(measurements: pd.DataFrame):
    """Return True for rows containing at least one anomaly."""
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return predict_column_anomalies(measurements, model).any(axis=1).to_numpy(dtype=bool)


def training_paths(training_dir: Path, training_percentages: tuple[float, ...] | None = None) -> list[Path]:
    """Select complete recordings by their scenario fault percentage."""
    paths = []
    for path in sorted(training_dir.iterdir()):
        if not path.is_dir():
            continue
        if training_percentages is not None:
            match = re.match(r"^(\d+)_(\d+)pct_", path.name)
            if match is None or float(f"{match[1]}.{match[2]}") not in training_percentages:
                continue
        paths.append(path / "measurements.parquet")
    if not paths:
        raise ValueError(f"No training recordings match percentages {training_percentages}")
    return paths


def train(training_percentages: tuple[float, ...] | None = None, model_path: Path | None = None) -> None:
    training_dir = REPO_ROOT / "data" / "training"

    paths = training_paths(training_dir, training_percentages)
    print("Training recordings: " + ", ".join(path.parent.name for path in paths))

    train_data = pd.concat(
        [pd.read_parquet(path) for path in paths],
        axis=0,
        ignore_index=True,
    )

    thresholds = {}
    columns = set(train_data.columns)
    state_groups = {}
    warn_missing_values(train_data, [column for column in train_data if is_ad_signal(column)])

    for column in train_data.columns:
        if not is_ad_signal(column):
            continue
        print(f"Processing column '{column}'")

        if train_data[column].notna().sum() == 0:
            continue

        state_column = state_column_for_signal(column, columns)
        if state_column is None:
            # Components without an exported state retain unconditional thresholds.
            thresholds[column] = learn_profile(train_data[column])
        else:
            if state_column not in state_groups:
                state_groups[state_column] = train_data.groupby(state_column, sort=False).indices
            profiles = {}
            for state, positions in state_groups[state_column].items():
                series = train_data[column].iloc[positions]
                if series.notna().any():
                    profiles[str(state)] = learn_profile(series)
            if profiles:
                thresholds[column] = {
                    "type": next(iter(profiles.values()))["type"],
                    "state_column": state_column,
                    "states": profiles,
                }

    output_path = model_path if model_path is not None else MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Saved model to {output_path}")


def learn_profile(series: pd.Series) -> dict:
    """Learn one numeric range or categorical set from the supplied state samples."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return {
            "type": "numeric",
            "lower": float(series.quantile(PERCENTILE_MARGIN)),
            "upper": float(series.quantile(1 - PERCENTILE_MARGIN)),
        }
    return {"type": "categorical", "categories": series.dropna().astype(str).unique().tolist()}


if __name__ == "__main__":
    train()
