"""Export an experimental mask on the original grid of a research head CT."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform
from torch import nn

from train_segmentation import SEGMENTERS, prepare_slices


@torch.inference_mode()
def predict_mask(scan, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    threshold = float(saved["threshold"])
    input_channels = saved.get("input_channels", 1)
    if saved["input_size"] != 128 or input_channels not in (1, 3) or not 0 < threshold < 1:
        raise ValueError("Unsupported segmentation checkpoint settings.")
    architecture = saved.get("architecture", "baseline")
    if architecture not in SEGMENTERS:
        raise ValueError("Unsupported segmentation architecture.")
    model = SEGMENTERS[architecture](input_channels)
    model.load_state_dict(saved["weights"])
    model.eval()
    canonical = nib.as_closest_canonical(scan)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise ValueError("Expected an orientable axial volume.")
    values = canonical.get_fdata(dtype=np.float32)
    slices = prepare_slices(values, input_channels)
    frames = []
    for batch in slices.split(8):
        scores = nn.functional.interpolate(
            model(batch).sigmoid(), size=(values.shape[1], values.shape[0]),
            mode="bilinear", align_corners=False,
        )
        frames.append((scores[:, 0] >= threshold).numpy())
    mask = np.concatenate(frames).transpose(2, 1, 0)
    restore = ornt_transform(io_orientation(canonical.affine), io_orientation(scan.affine))
    return np.ascontiguousarray(apply_orientation(mask, restore), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Research head CT NIfTI")
    parser.add_argument("output", type=Path, help="New .nii or .nii.gz mask file")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ct-segmenter.pt"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve any existing file.")
    if not args.output.name.lower().endswith((".nii", ".nii.gz")):
        parser.error("The mask filename must end in .nii or .nii.gz.")
    if not args.checkpoint.is_file():
        parser.error(f"No trained segmentation checkpoint found at {args.checkpoint}.")
    scan = nib.load(args.scan)
    mask = predict_mask(scan, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = nib.Nifti1Image(mask, scan.affine)
    result.header.set_xyzt_units(scan.header.get_xyzt_units()[0])
    nib.save(result, args.output)
    print(f"Saved experimental model mask with {int(mask.sum()):,} marked voxels to {args.output}")


if __name__ == "__main__":
    main()
