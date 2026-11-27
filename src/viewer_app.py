"""Interactive research viewer for one CT slice at a time."""

import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
from pydicom.pixels import apply_modality_lut
from PIL import Image, ImageDraw, ImageFilter

TEACHING_CT = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Intracerebral_hemorrhage.jpg"


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


def render_slice(values, center, width, inverted=False, mask=None, opacity=0.6, warm=False):
    lower = center - width / 2
    gray = np.clip((values - lower) / width, 0, 1)
    if inverted:
        gray = 1 - gray
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if mask is not None and mask.any() and opacity > 0:
        region = Image.fromarray(mask.astype(np.uint8) * 255)
        halo = np.asarray(region.filter(ImageFilter.GaussianBlur(5))) / 255
        tint = np.array([1.0, 0.31, 0.12] if warm else [0.0, 0.85, 1.0])
        outline = np.array([1.0, 0.83, 0.58] if warm else [0.65, 0.98, 1.0])
        rgb += (halo * opacity * 0.45)[..., None] * tint
        edge = mask & (np.asarray(region.filter(ImageFilter.MinFilter(3))) == 0)
        rgb[mask] = rgb[mask] * (1 - opacity) + tint * opacity
        rgb[edge] = rgb[edge] * (1 - opacity) + outline * opacity
        np.clip(rgb, 0, 1, out=rgb)
    rendered = (rgb * 255).astype(np.uint8)
    if warm and mask is not None and mask.any() and opacity > 0:
        # Center the guide on a marked pixel, including for disconnected masks.
        rows, cols = np.nonzero(mask)
        center_pixel = ((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2).argmin()
        y, x = int(rows[center_pixel]), int(cols[center_pixel])
        radius = max(7, round(min(mask.shape) * 0.047))
        base = Image.fromarray(rendered)
        guide = Image.new("RGBA", base.size)
        draw = ImageDraw.Draw(guide)
        ink = (255, 158, 116, round(230 * opacity))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=ink, width=2)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            draw.line((x + dx * (radius - 3), y + dy * (radius - 3),
                       x + dx * (radius + 8), y + dy * (radius + 8)), fill=ink, width=2)
        rendered = np.asarray(Image.alpha_composite(base.convert("RGBA"), guide).convert("RGB"))
    return rendered


def show_overlay_reveal(values, center, width, inverted, mask, opacity):
    if mask is None or not mask.any() or opacity <= 0:
        return
    if not st.button("Play region reveal"):
        return
    rows, cols = np.nonzero(mask)
    # Reveal outward from the marked pixel nearest the region's centroid.
    seed = ((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2).argmin()
    distance_squared = (rows - rows[seed]) ** 2 + (cols - cols[seed]) ** 2
    frames = [Image.fromarray(render_slice(values, center, width, inverted))]
    for fraction in np.linspace(0, 1, 16):
        visible = np.zeros_like(mask, dtype=bool)
        revealed = distance_squared <= distance_squared.max() * fraction**2
        visible[rows[revealed], cols[revealed]] = True
        frames.append(Image.fromarray(render_slice(values, center, width, inverted, visible, opacity)))
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:],
                   duration=[180] + [65] * 15 + [1000])
    st.image(buffer.getvalue(), caption="Animated reveal of the existing region. No time progression is shown.", width=560)


@st.cache_data(show_spinner="Opening annotated case…")
def read_annotated_case(scan_path, mask_path, scan_mtime, mask_mtime):
    import nibabel as nib

    return aligned_volumes(nib.load(scan_path), nib.load(mask_path))


def decode_nifti(contents):
    import nibabel as nib

    limit = 256 * 1024**2
    if contents.startswith(bytes((31, 139))):
        with gzip.GzipFile(fileobj=io.BytesIO(contents)) as compressed:
            contents = compressed.read(limit + 1)
    if len(contents) > limit:
        raise ValueError("Each uncompressed volume must be at most 256 MiB.")
    try:
        return nib.Nifti1Image.from_bytes(contents)
    except nib.wrapstruct.WrapStructError as error:
        raise ValueError("The file does not contain a valid NIfTI header.") from error


@st.cache_data(show_spinner="Opening uploaded case")
def read_uploaded_case(scan_bytes, mask_bytes):
    return aligned_volumes(decode_nifti(scan_bytes), decode_nifti(mask_bytes))


def aligned_volumes(scan, mask):
    import nibabel as nib

    scan, mask = nib.as_closest_canonical(scan), nib.as_closest_canonical(mask)
    if len(scan.shape) != 3 or scan.shape != mask.shape:
        raise ValueError("The scan and annotation must have matching 3D dimensions.")
    if not np.allclose(scan.affine, mask.affine, atol=0.01):
        raise ValueError("The scan and annotation must share the same spatial coordinates.")
    if nib.aff2axcodes(scan.affine) != ("R", "A", "S"):
        raise ValueError("The scan could not be oriented for axial viewing.")
    values = scan.get_fdata(dtype=np.float32)
    labels = mask.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(labels).all():
        raise ValueError("The scan and annotation must contain finite pixel values.")
    return values, labels > 0


def annotated_case():
    from dashboard import scan_montage

    try:
        import nibabel as nib
    except ModuleNotFoundError:
        st.info("The annotated case viewer needs nibabel. Install the project requirements to enable it.")
        return

    folder = Path(__file__).resolve().parents[1] / "data" / "annotated_case"
    scan, mask = folder / "scan.nii.gz", folder / "mask.nii.gz"
    st.subheader("CT volume & region overlay")
    st.caption("The orange overlay displays the supplied mask.")
    try:
        if scan.is_file() and mask.is_file():
            values, labels = read_annotated_case(
                str(scan), str(mask), scan.stat().st_mtime_ns, mask.stat().st_mtime_ns
            )
        else:
            with st.container(border=True):
                st.markdown("**Open a research case**")
                st.write("Choose a scan and annotation with the same case ID (.nii or .nii.gz).")
                left, right = st.columns(2)
                scan_upload = left.file_uploader("CT scan", type=["nii", "gz"], key="scan")
                mask_upload = right.file_uploader("Aligned region mask", type=["nii", "gz"], key="mask")
                st.link_button("Open PhysioNet dataset", "https://physionet.org/content/ct-ich/1.3.1/#files")
            if scan_upload is None or mask_upload is None:
                return
            values, labels = read_uploaded_case(scan_upload.getvalue(), mask_upload.getvalue())
    except (ValueError, OSError, EOFError, nib.filebasedimages.ImageFileError, nib.spatialimages.HeaderDataError) as error:
        st.error(f"Unable to load the annotated case. {error}")
        return
    display, metrics = st.columns([3.15, 1], gap="medium")
    with display, st.container(border=True):
        plane_controls = st.container()
        scans = st.container()
        controls = st.container()
    with plane_controls:
        plane = st.segmented_control(
            "Scan plane", ["Axial", "Coronal", "Sagittal"], default="Axial",
        ) or "Axial"
    with controls:
        axis = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[plane]
        counts = labels.sum(axis=tuple(i for i in range(3) if i != axis))
        initial = int(counts.argmax()) if counts.any() else len(counts) // 2
        index = (
            st.slider(f"{plane} slice", 1, len(counts), initial + 1, key=f"slice_{plane}") - 1
            if len(counts) > 1 else 0
        )
        show_overlay = st.toggle("Show supplied mask", value=True)
        opacity = st.slider("Overlay opacity", 0, 100, 65, disabled=not show_overlay) / 100
        compare = st.toggle("Compare side by side", disabled=not show_overlay)
        grid = st.toggle("Show neighboring slices", value=True, disabled=len(counts) < 2)
        st.caption("Each plane opens on the slice with the most marked pixels.")
    with scans:
        def to_display(volume, position=index):
            section = np.take(volume, position, axis=axis).T[::-1]
            return section[:, ::-1] if plane != "Sagittal" else section

        overlay = to_display(labels) if show_overlay else None
        axial = to_display(values)
        highlighted = render_slice(axial, 40, 120, mask=overlay, opacity=opacity, warm=True)
        if grid and len(counts) > 1:
            start = max(0, min(index - 1, len(counts) - 4))
            tiles = []
            for position in range(start, min(start + 4, len(counts))):
                image = render_slice(to_display(values, position), 40, 120,
                                     mask=to_display(labels, position) if show_overlay else None,
                                     opacity=opacity, warm=True)
                tiles.append((png_bytes(image), f"{plane.upper()} / SLICE {position + 1:03d}", position == index))
            scan_montage(tiles)
        elif compare and show_overlay:
            scan_montage([
                (png_bytes(render_slice(axial, 40, 120)), "ORIGINAL CT", False),
                (png_bytes(highlighted), "SUPPLIED MASK", True),
            ])
        else:
            scan_montage([(png_bytes(highlighted), f"{plane.upper()} / SLICE {index + 1:03d}", True)])
    with controls:
        show_overlay_reveal(axial, 40, 120, False, overlay, opacity)
    with metrics, st.container(border=True):
        st.subheader("Annotation metrics")
        st.metric("Marked pixels in selected slice", f"{int(counts[index]):,}")
        st.metric("Annotated voxels in volume", f"{np.count_nonzero(labels):,}")
        st.caption("Counts are computed from the supplied mask.")


@st.cache_data(show_spinner="Opening held-out case", max_entries=2)
def read_review_case(scan_path, mask_path, prediction_path, modified):
    import nibabel as nib

    scan = nib.load(scan_path)
    values, reference = aligned_volumes(scan, nib.load(mask_path))
    _, prediction = aligned_volumes(scan, nib.load(prediction_path))
    return values, reference, prediction


def model_review():
    st.subheader("Model review")
    st.caption(
        "Two selected BHSD test cases. Cyan shows the reference annotation. "
        "Orange shows the experimental single-slice model output."
    )
    case = st.segmented_control(
        "Review case", ["Strong overlap", "Missed region"], default="Strong overlap",
    ) or "Strong overlap"
    case_key = "strong" if case == "Strong overlap" else "failure"
    folder = Path(__file__).resolve().parents[1] / "data" / "model-review" / case_key
    paths = [folder / name for name in ("scan.nii.gz", "mask.nii.gz", "prediction.nii.gz")]
    if not all(path.is_file() for path in paths):
        st.info("This review case needs a local scan, reference mask, and saved model prediction.")
        return
    try:
        from nibabel.filebasedimages import ImageFileError
        from nibabel.spatialimages import HeaderDataError
    except ModuleNotFoundError:
        st.info("Install nibabel to open the review cases.")
        return
    try:
        values, reference, prediction = read_review_case(
            *(str(path) for path in paths), tuple(path.stat().st_mtime_ns for path in paths),
        )
    except (ValueError, OSError, EOFError, ImageFileError, HeaderDataError) as error:
        st.error(f"Unable to load the model review case. {error}")
        return

    slices = reference.sum(axis=(0, 1))
    case_metrics, controls = st.columns([1, 2], gap="medium")
    with case_metrics, st.container(border=True):
        overlap = int(np.count_nonzero(reference & prediction))
        reference_count = int(slices.sum())
        predicted_count = int(np.count_nonzero(prediction))
        total = reference_count + predicted_count
        st.metric("Whole-volume Dice", f"{2 * overlap / total:.3f}" if total else "No marked voxels")
    with controls, st.container(border=True):
        initial = int(slices.argmax()) + 1
        index = (
            st.slider("Axial slice", 1, len(slices), initial, key=f"review_slice_{case_key}") - 1
            if len(slices) > 1 else 0
        )
        opacity = st.slider("Overlay opacity", 0, 100, 70, key="review_opacity") / 100

    section = values[:, :, index].T[::-1, ::-1]
    ref_slice = reference[:, :, index].T[::-1, ::-1]
    pred_slice = prediction[:, :, index].T[::-1, ::-1]
    for column, (title, overlay, warm) in zip(st.columns(3), (
        ("Original CT", None, False),
        ("Reference annotation", ref_slice, False),
        ("Model prediction", pred_slice, True),
    )):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.image(render_slice(section, 40, 120, mask=overlay, opacity=opacity, warm=warm))
            if overlay is not None:
                st.caption(f"{np.count_nonzero(overlay):,} marked pixels on this slice")


def volume_prediction():
    st.subheader("Predict on a CT volume")
    st.caption("Use a de-identified axial head CT in NIfTI format. Experimental research output.")
    upload = st.file_uploader("Research CT volume (.nii or .nii.gz)", type=["nii", "gz"])
    project = Path(__file__).resolve().parents[1]
    example = project / "data" / "model-review" / "strong" / "scan.nii.gz"
    use_example = upload is None and st.toggle(
        "Use the BHSD research example", disabled=not example.is_file(),
    )
    checkpoint = project / "models" / "ct-segmenter-balanced-expanded.pt"
    if not checkpoint.is_file():
        checkpoint = project / "assets" / "checkpoints" / checkpoint.name
    if upload is None and not use_example:
        st.session_state.pop("volume_prediction", None)
        return
    if not checkpoint.is_file():
        st.info("Add the trained segmentation checkpoint to the models folder.")
        return

    contents = example.read_bytes() if use_example else upload.getvalue()
    scan_id = (hashlib.sha256(contents).hexdigest(), checkpoint.stat().st_mtime_ns)
    saved_id, result = st.session_state.get("volume_prediction", (None, None))
    if saved_id != scan_id:
        st.session_state.pop("volume_prediction", None)
        result = None

    if st.button("Generate experimental overlay", type="primary"):
        try:
            import nibabel as nib
            from nibabel.filebasedimages import ImageFileError
            from nibabel.spatialimages import HeaderDataError
            from predict_segmentation import predict_mask
        except ImportError as error:
            st.error(f"Segmentation setup is incomplete. {error}")
            return
        try:
            with st.spinner("Running segmentation on the CT volume"):
                scan = decode_nifti(contents)
                if len(scan.shape) != 3 or min(scan.shape) < 1:
                    raise ValueError("Choose a non-empty 3D CT volume.")
                mask = predict_mask(scan, checkpoint)
                result = aligned_volumes(scan, nib.Nifti1Image(mask, scan.affine))
                st.session_state["volume_prediction"] = (scan_id, result)
        except (ValueError, OSError, EOFError, RuntimeError, ImageFileError, HeaderDataError) as error:
            st.error(f"Unable to generate an overlay. {error}")
            return

    if result is None:
        return
    values, prediction = result
    counts = prediction.sum(axis=(0, 1))
    if not counts.any():
        st.info("The model produced an empty mask. A bleed may still be present.")
    selected = int(counts.argmax()) if counts.any() else len(counts) // 2
    controls, summary = st.columns([2, 1])
    with controls:
        index = (
            st.slider("Axial slice", 1, len(counts), selected + 1, key=f"prediction-slice-{scan_id}") - 1
            if len(counts) > 1 else 0
        )
        opacity = st.slider("Overlay opacity", 0, 100, 70, key="prediction-opacity") / 100
    with summary:
        st.metric("Model-marked pixels on this slice", f"{int(counts[index]):,}")
    section = values[:, :, index].T[::-1, ::-1]
    overlay = prediction[:, :, index].T[::-1, ::-1]
    original, predicted = st.columns(2)
    with original, st.container(border=True):
        st.markdown("**Original CT**")
        st.image(render_slice(section, 40, 120))
    with predicted, st.container(border=True):
        st.markdown("**Experimental model overlay**")
        st.image(render_slice(section, 40, 120, mask=overlay, opacity=opacity, warm=True))
    st.caption("Orange marks pixels selected by the model. Use the annotated review cases to see examples of missed regions.")

def landing():
    from title_page import title_page

    title_page(home_page)


def home():
    from title_page import workspace_entry
    from dashboard import brain_reference, dashboard_header

    workspace_entry()
    st.page_link(landing_page, label="Back to title", icon=":material/arrow_back:")
    dashboard_header()
    main_panel, hologram, preview = st.columns([1, 2.65, 1.2], gap="medium")
    with main_panel, st.container(border=True):
        st.subheader("Scan controls")
        st.caption("Use de-identified research files for uploads.")
        uploaded = st.file_uploader("CT slice", type=["dcm"], key="dashboard-dicom")
        if uploaded is not None:
            st.session_state["analysis_source"] = "slice"
            st.session_state["analysis_scan"] = (uploaded.name, uploaded.getvalue())
            st.switch_page(analysis_page)
        if st.button("Open teaching scan", type="primary"):
            st.session_state["analysis_source"] = "slice"
            st.session_state.pop("analysis_scan", None)
            st.switch_page(analysis_page)
        if st.button("Predict on a CT volume"):
            st.session_state["analysis_source"] = "volume-prediction"
            st.switch_page(analysis_page)
        if st.button("Review model results"):
            st.session_state["analysis_source"] = "review"
            st.switch_page(analysis_page)
        if st.button("Open volume + mask"):
            st.session_state["analysis_source"] = "volume"
            st.switch_page(analysis_page)
    with hologram, st.container(border=True):
        brain_reference()
    with preview, st.container(border=True):
        st.subheader("Axial head CT")
        st.image(TEACHING_CT)
        st.caption("Published teaching scan · Yadav et al. (2007), CC BY 2.0")
        st.markdown("[Image source](https://commons.wikimedia.org/wiki/File:Intracerebral_hemorrhage.jpg) · [License](https://creativecommons.org/licenses/by/2.0/)")

def analysis():
    from dashboard import dashboard_header, scan_montage, scan_workspace_style

    dashboard_header()
    st.page_link(home_page, label="Back to dashboard", icon=":material/arrow_back:")
    scan_workspace_style()
    if st.session_state.get("analysis_source") == "volume-prediction":
        volume_prediction()
        return
    if st.session_state.get("analysis_source") == "review":
        model_review()
        return
    if st.session_state.get("analysis_source") == "volume":
        annotated_case()
        return
    with st.expander("Open a different scan"):
        uploaded = st.file_uploader("Research CT slice (.dcm)", type=["dcm"], key="analysis-dicom")
        if st.button("Open CT volume and mask"):
            st.session_state["analysis_source"] = "volume"
            st.rerun()
    case = st.session_state.get("analysis_scan")
    if uploaded is not None:
        case = (uploaded.name, uploaded.getvalue())
        st.session_state["analysis_scan"] = case
    if case is None:
        image_col, details = st.columns([3.15, 1], gap="medium")
        with image_col, st.container(border=True):
            st.subheader("Axial head CT")
            st.image(TEACHING_CT, width=560)
            st.caption("Published teaching image · Yadav et al. (2007), CC BY 2.0")
            st.markdown("[Image source](https://commons.wikimedia.org/wiki/File:Intracerebral_hemorrhage.jpg) · [License](https://creativecommons.org/licenses/by/2.0/)")
        with details, st.container(border=True):
            st.subheader("Teaching example")
            st.write("The authors identify a thalamic hemorrhage on this head CT.")
            st.caption("The published JPEG is for visual reference. Upload a research DICOM to run the classifier.")
        return

    name, contents = case
    try:
        values, inverted = read_slice(contents)
        if not np.isfinite(values).all():
            raise ValueError("The slice contains invalid pixel values.")
    except (ValueError, OSError, RuntimeError, NotImplementedError, AttributeError) as error:
        st.error(f"Unable to open this slice. {error}")
        return

    model_dir = Path(__file__).resolve().parents[1] / "models"
    candidates = (
        model_dir / "ct-expanded-spatial.pt",
        model_dir.parent / "assets" / "checkpoints" / "ct-expanded-spatial.pt",
        model_dir / "ct-spatial-cnn.pt",
    )
    checkpoint = next((path for path in candidates if path.is_file()), candidates[-1])
    scan_id = (hashlib.sha256(contents).hexdigest(), checkpoint.name)
    image_panel, metrics = st.columns([3.15, 1], gap="medium")
    with image_panel, st.container(border=True):
        st.subheader("CT slice viewer")
        st.caption(f"{name} · Single DICOM image")
        scans = st.container()
        preset_col, center_col, width_col = st.columns(3)
        preset = preset_col.selectbox("Contrast preset", ["Brain", "Wide", "Bone"])
        center_default, width_default = {"Brain": (40, 80), "Wide": (50, 200), "Bone": (500, 2000)}[preset]
        center = center_col.slider("Window center", -200, 1000, center_default, key=f"analysis-{preset}-center")
        width = width_col.slider("Window width", 1, 3000, width_default, key=f"analysis-{preset}-width")
        mask = None
        with st.expander("Add a region overlay", expanded=True):
            st.caption("Load the mask for this exact slice, or open the volume predictor for a full CT.")
            if st.button("Open volume predictor"):
                st.session_state["analysis_source"] = "volume-prediction"
                st.rerun()
            annotation = st.file_uploader("Aligned binary mask (.npy)", type=["npy"], key=f"mask-{scan_id}")
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
            show_mask = st.toggle("Show mask", value=True, disabled=mask is None)
            opacity = st.slider("Overlay opacity", 0, 100, 65, disabled=mask is None) / 100
        image = render_slice(values, center, width, inverted, mask if show_mask else None, opacity, warm=True)
        with scans:
            preview = png_bytes(image)
            label = "SLICE 001 · MASK OVERLAY" if show_mask and mask is not None and mask.any() else "DICOM / SLICE 001"
            scan_montage([(preview, label, True)])
        st.download_button("Export slice PNG", preview, "ct-slice.png", "image/png")

    with metrics, st.container(border=True):
        st.subheader("Analysis metrics")
        st.caption("Experimental model scores for this slice")
        saved_id, scores = st.session_state.get("analysis_scores", (None, None))
        if saved_id != scan_id:
            scores = None
        if st.button("Run research classifier", type="primary"):
            try:
                from research_model import load_model, score_slice

                if not checkpoint.is_file():
                    st.info("Add the trained checkpoint at models/ct-spatial-cnn.pt.")
                else:
                    scores = score_slice(load_model(checkpoint), contents)
                    st.session_state["analysis_scores"] = (scan_id, scores)
            except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
                st.error(f"Classifier unavailable. {error}")
        if scores is not None:
            score = float(scores["any"])
            st.markdown(
                f'<div style="margin:18px auto;width:112px;height:112px;border-radius:50%;'
                f'background:conic-gradient(#68e8ff {score * 360:.1f}deg,#263b49 0);'
                'display:grid;place-items:center"><b style="background:#0b1a26;'
                'border-radius:50%;width:94px;height:94px;display:grid;place-items:center;'
                f'font-size:1.45rem">{score:.3f}</b></div>',
                unsafe_allow_html=True,
            )
            st.caption("Any hemorrhage · raw score")
            st.markdown("**Subtype scores**")
            for label, value in scores.items():
                if label != "any":
                    st.progress(value, text=f"{label.capitalize()} · {value:.3f}")
        st.divider()
        st.markdown("**Annotation**")
        if mask is None:
            st.caption("Upload an aligned mask to show the marked region on the scan.")
        else:
            st.metric("Marked pixels", f"{np.count_nonzero(mask):,}")


def main():
    global landing_page, home_page, analysis_page
    st.set_page_config(page_title="CT Slice Lab", layout="wide")
    landing_page = st.Page(landing, title="CT Hemorrhage", default=True)
    home_page = st.Page(home, title="Research workspace", url_path="workspace")
    analysis_page = st.Page(analysis, title="Scan analysis", url_path="analysis")
    st.navigation([landing_page, home_page, analysis_page], position="hidden").run()


def png_bytes(image):
    from matplotlib import image as mpl_image

    buffer = io.BytesIO()
    mpl_image.imsave(buffer, image, format="png")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
