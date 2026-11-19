"""Quickly inspect an RSNA-style image-level label CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUBTYPES = (
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
)


def inspect_csv(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"ID", "Label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)}; missing {sorted(missing)}. "
            f"Found: {list(frame.columns)}"
        )

    print(f"Rows: {len(frame):,}")
    print(f"Unique label IDs: {frame['ID'].nunique():,}")
    print("\nLabels by subtype (positive rows):")
    for subtype in SUBTYPES:
        mask = frame["ID"].astype(str).str.endswith(f"_{subtype}")
        count = int((frame.loc[mask, "Label"] == 1).sum())
        print(f"  {subtype:18s} {count:>10,}")

    print(
        "\nThese are image-level label counts, not counts of unique patients or exams."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to train.csv")
    args = parser.parse_args()
    inspect_csv(args.csv)


if __name__ == "__main__":
    main()
