"""Train a research slice classifier from a local RSNA subset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms.functional import normalize, resize

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def prepare_labels(csv_path, dicom_dir):
    frame = pd.read_csv(csv_path, usecols=["ID", "Label"])
    frame[["image", "subtype"]] = frame.ID.str.rsplit("_", n=1, expand=True)
    paths = {path.stem: path for path in dicom_dir.glob("*.dcm")}
    frame = frame[frame.image.isin(paths)].drop_duplicates(["image", "subtype", "Label"])
    if frame.empty or frame.duplicated(["image", "subtype"]).any():
        raise ValueError("No matching labels or conflicting duplicate labels.")
    labels = frame.pivot(index="image", columns="subtype", values="Label").reindex(columns=LABELS)
    if not labels.isin([0, 1]).all().all():
        raise ValueError("Every selected slice needs six binary labels.")
    patient_ids = []
    for image in labels.index:
        header = pydicom.dcmread(paths[image], stop_before_pixels=True)
        patient_id = str(header.get("PatientID", "")).strip()
        if not patient_id:
            raise ValueError(f"Missing PatientID in {paths[image].name}.")
        patient_ids.append(patient_id)
    return labels, paths, np.array(patient_ids)


class Slices(Dataset):
    def __init__(self, labels, paths):
        self.images = [paths[image] for image in labels.index]
        self.labels = torch.tensor(labels.to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        scan = pydicom.dcmread(self.images[index])
        pixels = scan.pixel_array.astype(np.float32)
        if scan.get("Modality") != "CT" or pixels.ndim != 2:
            raise ValueError(f"Expected one grayscale CT slice: {self.images[index].name}")
        hu = pixels * float(scan.RescaleSlope) + float(scan.RescaleIntercept)
        channels = [
            np.clip((hu - (level - width / 2)) / width, 0, 1)
            for level, width in [(40, 80), (80, 200), (40, 380)]
        ]
        image = resize(torch.from_numpy(np.stack(channels)), [224, 224], antialias=True)
        return normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), self.labels[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("dicom_dir", type=Path, help="Folder containing the selected DICOM slices")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/rsna_resnet50.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    torch.manual_seed(0)
    labels, paths, patients = prepare_labels(args.labels, args.dicom_dir)
    train, val = next(GroupShuffleSplit(test_size=0.15, random_state=0).split(labels, groups=patients))
    print(f"Train: {len(train)} slices, {len(set(patients[train]))} patients")
    print(f"Validation: {len(val)} slices, {len(set(patients[val]))} patients")
    print("Training positives:", labels.iloc[train].sum().to_dict())
    train_loader = DataLoader(Slices(labels.iloc[train], paths), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Slices(labels.iloc[val], paths), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        print(f"Epoch {epoch + 1}: training loss {total / len(train):.4f}")
    model.eval()
    with torch.inference_mode():
        scores = torch.cat([model(images.to(device)).sigmoid().cpu() for images, _ in val_loader]).numpy()
    for column, subtype in enumerate(LABELS):
        truth = labels.iloc[val][subtype].to_numpy()
        probabilities = scores[:, column].astype(np.float64)
        auc = f"{roc_auc_score(truth, probabilities):.4f}" if np.unique(truth).size == 2 else "undefined"
        print(f"{subtype}: positives={truth.sum()}, log loss={log_loss(truth, probabilities, labels=[0, 1]):.4f}, AUC={auc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "labels": LABELS}, args.output)


if __name__ == "__main__":
    main()
