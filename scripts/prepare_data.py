"""Prepare raw compressed simulation exports for training.

Run this script from the ``scripts`` directory:

    python prepare_data.py

It reads up to 1,000 JSONL samples from each ``.json.zst`` file in
``../data/training`` and ``../data/test``. For every input, it creates a
same-named directory containing ``measurements.parquet``, ``faults.parquet``,
and ``parameters.parquet``. Each parquet file uses ``simulationTime`` as its
index.
"""

import sys
import traceback
from pathlib import Path
from utils import (
    EXCLUDED_COLUMNS,
    EXCLUDED_COLUMN_PREFIXES,
    read_json_zst,
    split_dataframe,
)

dirs_to_transform = ["../data/training", "../data/test"]
for dir_tr in dirs_to_transform:
    input_files = sorted(
        path
        for path in Path(dir_tr).iterdir()
        if path.is_file() and path.name.endswith(".json.zst")
    )

    processed_files = 0
    failed_files = 0

    for input_file in input_files:
        try:
            output_dir = input_file.with_name(input_file.name.removesuffix(".json.zst"))
            df = read_json_zst(input_file, max_samples=1000)
            df = df.loc[
                :,
                [
                    column
                    for column in df.columns
                    if column not in EXCLUDED_COLUMNS
                    and not column.startswith(EXCLUDED_COLUMN_PREFIXES)
                ],
            ]
            output_dir.mkdir(exist_ok=True)
            split_dataframe(df, output_dir)
            print(f"Saved prepared parquet files to {output_dir}")
            processed_files += 1
        except Exception as error:
            failed_files += 1
            print(
                f"Failed to process {input_file}: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            traceback.print_exc()

    print(
        f"Finished processing {processed_files} of {len(input_files)} files "
        f"({failed_files} failed)."
    )
