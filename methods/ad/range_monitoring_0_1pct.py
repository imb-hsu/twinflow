"""Range monitoring trained on the 0% and 1% fault recordings."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from methods.ad import range_monitoring

METHOD_NAME = "Range Monitoring (0% + 1%)"
MODEL_PATH = REPO_ROOT / "models" / "range_monitoring_0_1pct.json"


def train():
    range_monitoring.train(training_percentages=(0, 1), model_path=MODEL_PATH)


def predict(measurements):
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return range_monitoring.predict_column_anomalies(measurements, model).any(axis=1).to_numpy(dtype=bool)


if __name__ == "__main__":
    train()
