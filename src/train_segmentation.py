"""Train an experimental pixel mask model on paired, annotated head CT volumes."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class Segmenter(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.detail = nn.Sequential(nn.Conv2d(input_channels, 16, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.output = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, image):
        detail = self.detail(image)
        context = nn.functional.interpolate(self.context(detail), size=detail.shape[-2:],
                                            mode="bilinear", align_corners=False)
        return self.output(torch.cat((detail, context), dim=1))


def prepare_slices(values, input_channels=1):
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Expected a 3D CT volume with finite intensities.")
    images = torch.from_numpy(np.clip((values.astype(np.float32) + 20) / 120, 0, 1)
                              .transpose(2, 1, 0).copy())[:, None]
    images = nn.functional.interpolate(images, (128, 128), mode="bilinear", align_corners=False)
    if input_channels == 3:
        previous = torch.cat((images[:1], images[:-1]), dim=0)
        following = torch.cat((images[1:], images[-1:]), dim=0)
        images = torch.cat((previous, images, following), dim=1)
    return images


def read_case(scan_path, mask_path, input_channels=1):
    scan, mask = map(lambda p: nib.as_closest_canonical(nib.load(p)), (scan_path, mask_path))
    if len(scan.shape) != 3 or scan.shape != mask.shape or not np.allclose(scan.affine, mask.affine, atol=0.01):
        raise ValueError(f"Scan and mask must share a 3D grid: {scan_path}")
    if nib.aff2axcodes(scan.affine) != ("R", "A", "S"):
        raise ValueError(f"Expected an orientable axial volume: {scan_path}")
    values, marked = (im.get_fdata(dtype=np.float32) for im in (scan, mask))
    if not np.isfinite(values).all() or not np.isin(marked, [0, 1]).all():
        raise ValueError(f"Expected finite CT intensities and binary masks: {scan_path}")
    labels = torch.from_numpy(marked.transpose(2, 1, 0).copy())[:, None]
    return (
        prepare_slices(values, input_channels),
        nn.functional.interpolate(labels, (128, 128), mode="nearest"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV columns: patient_id,scan,mask")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("models/ct-segmenter.pt"))
    parser.add_argument("--neighboring-slices", action="store_true",
                        help="Use the previous and next axial slices as additional inputs.")
    args = parser.parse_args()
    input_channels = 3 if args.neighboring_slices else 1
    if args.epochs < 1:
        parser.error("Epochs must be positive.")
    with args.manifest.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not all((row.get(key) or "").strip() for key in ("patient_id", "scan", "mask")) for row in rows):
        parser.error("Provide patient_id, scan, and mask for every case.")
    paths = [(args.manifest.parent / r["scan"], args.manifest.parent / r["mask"]) for r in rows]
    if len({scan.resolve() for scan, _ in paths}) != len(paths):
        parser.error("Duplicate scan paths are not allowed.")
    patients = [r["patient_id"].strip() for r in rows]
    if len(set(patients)) < 5:
        parser.error("Provide at least five distinct patients for a training and validation split.")
    train, validation = next(GroupShuffleSplit(test_size=.2, random_state=26).split(rows, groups=patients))
    cases = [read_case(*pair, input_channels) for pair in paths]
    images, labels = (torch.cat([cases[i][j] for i in train]) for j in (0, 1))
    if not labels.any() or not any(cases[i][1].any() for i in validation):
        parser.error("Both splits need examples with marked pixels.")
    torch.manual_seed(26)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Segmenter(input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    marked = labels.flatten(1).any(dim=1)
    marked_count = int(marked.sum())
    unmarked_count = len(marked) - marked_count
    sampling_weights = torch.where(
        marked, 1 / marked_count, 1 / max(1, unmarked_count),
    )
    sampler = WeightedRandomSampler(sampling_weights, len(marked), replacement=True)
    positive = labels[marked].sum().item()
    # Balance pixels using the foreground prevalence expected under this sampler.
    expected_prevalence = positive / (marked_count * labels[0].numel())
    if unmarked_count:
        expected_prevalence /= 2
    positive_weight = torch.tensor(
        min(30.0, max(1.0, ((1 - expected_prevalence) / expected_prevalence) ** 0.5)), device=device,
    )
    loader = DataLoader(TensorDataset(images, labels), batch_size=16, sampler=sampler)
    print(f"{len(set(np.asarray(patients)[train]))} training groups, "
          f"{len(set(np.asarray(patients)[validation]))} validation groups", flush=True)
    print(f"{marked_count}/{len(marked)} original training slices contain a mask. "
          "Sampler gives equal total weight to marked and unmarked slices when both exist.",
          flush=True)
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            p = logits.sigmoid()
            dice_loss = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            pixel_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=positive_weight,
            )
            (pixel_loss + dice_loss).backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
    model.eval()
    thresholds = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    dice = {threshold: [] for threshold in thresholds}
    negative_false_positive = {threshold: [] for threshold in thresholds}
    with torch.inference_mode():
        for i in validation:
            x, target = cases[i]
            target = target.bool()
            target_count = int(target.sum())
            scores = torch.cat([model(batch.to(device)).sigmoid().cpu() for batch in x.split(16)])
            for threshold in thresholds:
                pred = scores >= threshold
                if target_count:
                    dice[threshold].append(float(2 * (pred & target).sum() / (pred.sum() + target_count)))
                else:
                    negative_false_positive[threshold].append(bool(pred.any()))

    mean_dice = {threshold: float(np.mean(values)) for threshold, values in dice.items()}
    selected = max(thresholds, key=lambda t: (
        mean_dice[t], -sum(negative_false_positive[t]), -abs(t - 0.5),
    ))
    for threshold in dict.fromkeys((0.5, selected)):
        negatives = negative_false_positive[threshold]
        print(f"Threshold {threshold:.2f}: mean Dice {mean_dice[threshold]:.3f} "
              f"on {len(dice[threshold])} positive validation volumes at 128x128, "
              f"negative volumes with predicted pixels {sum(negatives)}/{len(negatives)}",
              flush=True)
        print("Positive volume Dice: " + ", ".join(f"{score:.3f}" for score in dice[threshold]),
              flush=True)
    print(f"Selected threshold {selected:.2f} using validation volumes", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weights": model.cpu().state_dict(),
        "input_size": 128,
        "input_channels": input_channels,
        "threshold": selected,
    }, args.output)


if __name__ == "__main__":
    main()
