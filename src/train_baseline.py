"""Train and evaluate a research-only CT bleed classification baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def features(path):
    scan = pydicom.dcmread(path)
    if scan.get("Modality") != "CT" or scan.get("RescaleType", "HU") != "HU":
        raise ValueError(f"Expected CT intensities in Hounsfield units: {path}")
    pixels = apply_modality_lut(scan.pixel_array, scan)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError(f"Expected a finite grayscale slice: {path}")
    # Average the relative amount of tissue in each intensity band.
    bins = [-np.inf, -100, 0, 20, 40, 60, 80, 100, 200, np.inf]
    return np.histogram(pixels, bins=bins)[0] / pixels.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "exam_id": str, "path": str})
    required = ["path", "patient_id", "exam_id", "label", "split"]
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        parser.error("Manifest needs complete path, patient_id, exam_id, label, split columns.")
    if not frame.label.isin([0, 1]).all() or set(frame.split) != {"train", "test"}:
        parser.error("Use binary slice labels and both train and test splits.")
    if (frame.groupby("patient_id").split.nunique() > 1).any():
        parser.error("A patient crosses the train/test split.")
    if (frame.groupby("exam_id").patient_id.nunique() > 1).any():
        parser.error("Exam IDs must belong to a single patient.")
    paths = [(args.manifest.parent / path).resolve() for path in frame.path]
    if len(set(paths)) != len(paths):
        parser.error("Each slice file must appear only once.")
    train = frame.split.eq("train").to_numpy()
    if frame.label[train].nunique() != 2:
        parser.error("Training requires positive and negative slices.")
    exams = frame.loc[~train].groupby("exam_id").label.max()
    if exams.nunique() != 2:
        parser.error("Testing requires positive and negative exams.")
    matrix = np.stack([features(path) for path in paths])
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(matrix[train], frame.label[train])
    predictions = frame.loc[~train, ["exam_id", "label"]].copy()
    predictions["probability"] = model.predict_proba(matrix[~train])[:, 1]
    # An exam is positive if any slice is positive. Max score is a ranking heuristic.
    predictions = predictions.groupby("exam_id", as_index=False).max()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    print(json.dumps({
        "held_out_exams": len(predictions),
        "auroc": roc_auc_score(predictions.label, predictions.probability),
        "average_precision": average_precision_score(predictions.label, predictions.probability),
    }, indent=2))


if __name__ == "__main__":
    main()
