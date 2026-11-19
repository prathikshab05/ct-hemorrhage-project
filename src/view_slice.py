"""Preview one CT slice from a research DICOM file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pydicom
from pydicom.pixels import apply_modality_lut


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicom", type=Path, help="Path to one CT DICOM file")
    parser.add_argument("--center", type=float, default=40)
    parser.add_argument("--width", type=float, default=80)
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("--width must be positive")

    scan = pydicom.dcmread(args.dicom)
    if scan.get("Modality") != "CT":
        parser.error("Expected a CT DICOM file")
    pixels = scan.pixel_array
    if pixels.ndim != 2:
        parser.error("Expected a single grayscale slice")

    intensities = apply_modality_lut(pixels, scan)
    half_width = args.width / 2
    figure, axes = plt.subplots()
    axes.imshow(
        intensities,
        cmap="gray_r" if scan.get("PhotometricInterpretation") == "MONOCHROME1" else "gray",
        vmin=args.center - half_width,
        vmax=args.center + half_width,
    )
    axes.set_title("CT slice preview")
    axes.set_axis_off()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
