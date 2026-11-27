"""Load the Kaggle spatial CNN and score one research DICOM slice."""

import io
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.pixels import apply_modality_lut
from torch import nn

LABELS = ("any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural")
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
EXPANDED_CHECKPOINT = MODEL_DIR / "ct-expanded-spatial.pt"
CHECKPOINT = EXPANDED_CHECKPOINT if EXPANDED_CHECKPOINT.is_file() else MODEL_DIR / "ct-spatial-cnn.pt"


def load_model(path=CHECKPOINT):
    model = nn.Sequential(
        nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(576, 6),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("The checkpoint has an unexpected label order.")
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


@torch.inference_mode()
def score_slice(model, contents):
    scan = pydicom.dcmread(io.BytesIO(contents))
    if scan.get("Modality") != "CT":
        raise ValueError("Select a CT DICOM slice.")
    hu = apply_modality_lut(scan.pixel_array, scan).astype(np.float32)
    if hu.ndim != 2 or not np.isfinite(hu).all():
        raise ValueError("Select one grayscale CT slice with finite intensities.")
    channels = np.stack([
        np.clip((hu - (center - width / 2)) / width, 0, 1)
        for center, width in [(40, 80), (80, 200), (40, 380)]
    ])
    image = nn.functional.interpolate(
        torch.from_numpy(channels)[None], (96, 96), mode="bilinear", align_corners=False,
    )
    return dict(zip(LABELS, model(image).sigmoid()[0].tolist()))
