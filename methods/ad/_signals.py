"""Shared signal policy for anomaly detectors."""

import warnings


def missing_value_warning(measurements, columns) -> str | None:
    counts = measurements[list(columns)].isna().sum()
    counts = counts[counts > 0]
    if counts.empty:
        return None
    rows = int(measurements[counts.index].isna().any(axis=1).sum())
    details = "; ".join(f"{column}: {int(count):,}" for column, count in counts.items())
    return (f"Warning: missing measurements in {rows:,} samples across {len(counts)} signals. "
            f"Missing values are ignored for anomaly scoring. Missing counts: {details}.")


def warn_missing_values(measurements, columns) -> None:
    message = missing_value_warning(measurements, columns)
    if message:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def is_ad_signal(column) -> bool:
    """Exclude CurrentPosition for every component and axis."""
    return str(column).rsplit(".", 1)[-1].casefold() != "currentposition"


def validate_autoencoder_columns(columns) -> None:
    if any(not is_ad_signal(column) for column in columns):
        raise ValueError(
            "Autoencoder model includes excluded CurrentPosition signals. "
            "Run python methods/ad/autoencoder.py to retrain it."
        )
