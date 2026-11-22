"""Evaluate one probability and reference label per held-out CT exam."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with exam_id, label, probability")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold fixed on validation data")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be in [0, 1].")
    frame = pd.read_csv(args.csv, dtype={"exam_id": str})
    if not {"exam_id", "label", "probability"}.issubset(frame):
        parser.error("Expected exam_id, label, and probability columns.")
    if frame.exam_id.isna().any() or frame.exam_id.duplicated().any():
        parser.error("Each row must have a unique exam ID.")
    labels = frame.label
    scores = pd.to_numeric(frame.probability, errors="coerce")
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or not scores.between(0, 1).all():
        parser.error("Both binary classes and finite probabilities in [0, 1] are required.")
    tn, fp, fn, tp = map(int, confusion_matrix(labels, scores >= args.threshold, labels=[0, 1]).ravel())
    print(json.dumps({
        "exams": len(frame),
        "positive_exams": tp + fn,
        "threshold": args.threshold,
        "auroc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "false_positives": fp,
        "false_negatives": fn,
    }, indent=2))


if __name__ == "__main__":
    main()
