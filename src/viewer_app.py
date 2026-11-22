"""Interactive research viewer for one CT slice at a time."""

import io

import numpy as np
import pydicom
import streamlit as st
from pydicom.pixels import apply_modality_lut


def demo_slice():
    """Create a stylized teaching image and an aligned example mask."""
    y, x = np.mgrid[-1:1:320j, -1:1:320j]
    radius = (x / 0.78) ** 2 + (y / 0.91) ** 2
    head = np.full(x.shape, -1000.0)
    head[radius < 1] = 24
    head[(radius > 0.78) & (radius < 0.94)] = 850
    tissue = radius <= 0.78
    texture = 4 * np.cos(32 * x + 6 * np.sin(9 * y)) * np.sin(19 * y)
    head[tissue] = (34 + texture)[tissue]
    ventricles = ((np.abs(x) - 0.09) / 0.06) ** 2 + ((y + 0.01) / 0.23) ** 2 < 1
    head[ventricles] = 8
    mask = ((x - 0.38 + 0.3 * (y + 0.19)) / 0.09) ** 2 + ((y + 0.19) / 0.15) ** 2 < 1
    head[mask] = 64
    return head, mask


def read_slice(contents):
    scan = pydicom.dcmread(io.BytesIO(contents))
    if scan.get("Modality") != "CT":
        raise ValueError("Choose a CT DICOM file.")
    pixels = scan.pixel_array
    if pixels.ndim != 2:
        raise ValueError("Choose a single grayscale CT slice.")
    return apply_modality_lut(pixels, scan), scan.get("PhotometricInterpretation") == "MONOCHROME1"


def render_slice(values, center, width, inverted=False, mask=None, opacity=0.6):
    lower = center - width / 2
    gray = np.clip((values - lower) / width, 0, 1)
    if inverted:
        gray = 1 - gray
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mask is not None:
        amber = np.array([1.0, 0.55, 0.18])
        rgb[mask] = rgb[mask] * (1 - opacity) + amber * opacity
    return (rgb * 255).astype(np.uint8)


def main():
    st.set_page_config(page_title="CT Slice Lab", layout="wide")
    st.title("CT Slice Lab")
    st.caption("Research viewer for head CT slices")
    controls, display = st.columns([1, 2], gap="large")

    with controls:
        st.subheader("Image controls")
        mode = st.radio("Source", ["Demo slice", "Research DICOM"])
        uploaded = st.file_uploader("Single CT slice (.dcm)", type=["dcm"]) if mode == "Research DICOM" else None
        annotation = (
            st.file_uploader("Optional aligned binary mask (.npy)", type=["npy"])
            if uploaded is not None else None
        )
        preset = st.selectbox("Window preset", ["Brain", "Wide", "Bone"])
        default_center, default_width = {"Brain": (40, 80), "Wide": (50, 200), "Bone": (500, 2000)}[preset]
        center = st.slider("Window center", -200, 1000, default_center, key=f"{preset}_center")
        width = st.slider("Window width", 1, 3000, default_width, key=f"{preset}_width")
        has_mask = mode == "Demo slice" or annotation is not None
        show_mask = has_mask and st.toggle("Show region overlay", value=True)
        opacity = st.slider("Highlight opacity", 0, 100, 65) / 100 if show_mask else 0

    with display:
        if mode == "Demo slice":
            st.subheader("Simulated axial slice")
            st.caption("The amber region is an illustrative mask on synthetic data.")
            values, mask = demo_slice()
            inverted = False
        else:
            st.subheader("Uploaded CT slice")
            if uploaded is None:
                st.info("Choose a DICOM slice from your research dataset.")
                return
            try:
                values, inverted = read_slice(uploaded.getvalue())
            except (ValueError, OSError, RuntimeError, NotImplementedError, AttributeError) as error:
                st.error(f"Unable to open this slice. {error}")
                return
            mask = None
            if annotation is not None:
                try:
                    candidate = np.load(io.BytesIO(annotation.getvalue()), allow_pickle=False)
                    if not isinstance(candidate, np.ndarray) or candidate.shape != values.shape:
                        raise ValueError("Mask dimensions must match the CT pixel grid.")
                    if not np.isin(candidate, [0, 1]).all():
                        raise ValueError("Mask pixels must be 0 or 1.")
                    mask = candidate.astype(bool)
                except (ValueError, OSError, EOFError) as error:
                    st.warning(f"Overlay unavailable. {error}")
                if mask is not None:
                    st.caption("Uploaded annotation. Confirm that it belongs to this slice.")
        image = render_slice(values, center, width, inverted, mask if show_mask else None, opacity)
        st.image(image, width=560)
        st.download_button(
            "Download displayed slice as PNG",
            png_bytes(image),
            "ct-slice.png",
            "image/png",
        )


def png_bytes(image):
    from matplotlib import image as mpl_image

    buffer = io.BytesIO()
    mpl_image.imsave(buffer, image, format="png")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
