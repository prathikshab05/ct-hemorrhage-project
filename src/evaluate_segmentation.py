"""Measure experimental segmentation overlap on held-out native CT grids."""

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import nibabel as nib
import numpy as np

from predict_segmentation import predict_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Held-out CSV with patient_id,scan,mask")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not all((row.get(key) or "").strip() for key in
                               ("patient_id", "scan", "mask")) for row in rows):
        parser.error("Provide an ID, scan, and mask for each held-out case.")

    dice, precision, recall = [], [], []
    negative_count = false_positive_count = 0
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["case", "reference_voxels", "predicted_voxels", "dice", "precision", "recall"])
    for index, row in enumerate(rows, start=1):
        scan = nib.load(args.manifest.parent / row["scan"])
        mask = nib.load(args.manifest.parent / row["mask"])
        if len(scan.shape) != 3 or scan.shape != mask.shape or not np.allclose(
            scan.affine, mask.affine, atol=0.01,
        ):
            parser.error(f"Case {index} does not have an aligned 3D reference mask.")
        reference = np.asanyarray(mask.dataobj)
        if not np.isin(reference, [0, 1]).all():
            parser.error(f"Case {index} does not have a binary reference mask.")
        target = reference.astype(bool)
        predicted = predict_mask(scan, args.checkpoint).astype(bool)
        reference_count = int(target.sum())
        predicted_count = int(predicted.sum())
        if reference_count:
            overlap = int((predicted & target).sum())
            dice.append(2 * overlap / (reference_count + predicted_count))
            precision.append(overlap / predicted_count if predicted_count else 0.0)
            recall.append(overlap / reference_count)
            writer.writerow([index, reference_count, predicted_count,
                             f"{dice[-1]:.4f}", f"{precision[-1]:.4f}", f"{recall[-1]:.4f}"])
        else:
            negative_count += 1
            false_positive_count += bool(predicted_count)
            writer.writerow([index, reference_count, predicted_count, "", "", ""])

    print(f"Positive volumes: {len(dice)}")
    if dice:
        print(f"Mean native-grid Dice: {mean(dice):.4f}")
        print(f"Mean voxel precision: {mean(precision):.4f}")
        print(f"Mean voxel recall: {mean(recall):.4f}")
    print(f"Negative volumes with predicted voxels: {false_positive_count}/{negative_count}")


if __name__ == "__main__":
    main()
