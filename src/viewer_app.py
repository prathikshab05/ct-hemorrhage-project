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
    from dashboard import scan_montage, scan_workspace_style

    scan_workspace_style(show_heading=False)
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
    from title_page import workspace_entry

    workspace_entry()
    st.html(Path(__file__).with_name("scan_viewer.css"))
    st.html("""
        <div class="ct-viewer-rail">
            <span>CT HEMORRHAGE</span>
            <span>SCIENTIFIC PROTOTYPE</span>
        </div>
    """)
    back, onward = st.columns(2)
    back.page_link(analysis_page, label="Back to scan viewer", icon=":material/arrow_back:")
    onward.page_link(anatomy_route, label="Explore anatomy", icon=":material/arrow_forward:")
    st.title("Model results")
    st.caption("Experimental hemorrhage segmentation · Research use only. Not for diagnostic purposes.")

    active_id, active_scan = st.session_state.get("viewer_volume", (None, None))
    has_active_scan = active_scan is not None and st.session_state.get("analysis_source") == "volume-prediction"
    options = ["Strong overlap", "Missed region"]
    if has_active_scan:
        options.insert(0, "Current CT volume")
    tabs = st.tabs(options, key="results-case-tab", on_change="rerun")
    for case, tab in zip(options, tabs):
        if tab.open:
            with tab:
                render_comparison(case, active_id, active_scan)

    with st.container(border=True, key="ct-results-metrics"):
        st.subheader("Held-out evaluation")
        dice, precision, recall = st.columns(3)
        dice.metric("Mean Dice", "0.354")
        precision.metric("Mean voxel precision", "40.9%")
        recall.metric("Mean voxel recall", "39.4%")
        st.caption("Eight BHSD volumes with annotated hemorrhage · Metrics averaged per volume. "
                   "Scan-level specificity was not measured on this set.")


@st.cache_data(show_spinner="Loading the published BHSD example and running segmentation", max_entries=2)
def read_published_review(case_key, checkpoint, checkpoint_modified):
    import nibabel as nib

    from predict_segmentation import predict_mask
    from review_examples import fetch_case

    scan_bytes, reference_bytes = fetch_case(case_key)
    scan = nib.as_closest_canonical(decode_nifti(scan_bytes))
    values, reference = aligned_volumes(scan, decode_nifti(reference_bytes))
    prediction = predict_mask(scan, checkpoint).astype(bool)
    return values, reference, prediction


def render_comparison(case, active_id, active_scan):
    from nibabel.filebasedimages import ImageFileError
    from nibabel.wrapstruct import WrapStructError
    from nibabel.spatialimages import HeaderDataError
    from review_examples import CASES

    values = reference = prediction = None
    if case == "Current CT volume":
        scan, values, _ = active_scan
        saved_id, saved_mask = st.session_state.get("volume_prediction", (None, None))
        if saved_id is not None and saved_id[0] == active_id:
            prediction = saved_mask
        uploaded = st.file_uploader("Optional aligned NIfTI reference mask", type=["nii", "gz"],
                                    key=f"results-mask-{active_id}")
        if uploaded is not None:
            try:
                _, reference = aligned_volumes(scan, decode_nifti(uploaded.getvalue()))
            except (ValueError, OSError, EOFError, ImageFileError, WrapStructError, HeaderDataError) as error:
                st.error(f"Unable to open the reference mask. {error}")
    else:
        case_key = "strong" if case == "Strong overlap" else "failure"
        folder = Path(__file__).resolve().parents[1] / "data" / "model-review" / case_key
        paths = [folder / name for name in ("scan.nii.gz", "mask.nii.gz", "prediction.nii.gz")]
        try:
            if all(path.is_file() for path in paths):
                values, reference, prediction = read_review_case(
                    *(str(path) for path in paths), tuple(path.stat().st_mtime_ns for path in paths),
                )
            else:
                checkpoint = Path(__file__).resolve().parents[1] / "assets/checkpoints/ct-segmenter-balanced-expanded.pt"
                values, reference, prediction = read_published_review(case_key, str(checkpoint), checkpoint.stat().st_mtime_ns)
        except (ValueError, OSError, EOFError, RuntimeError, ImageFileError, WrapStructError, HeaderDataError) as error:
            st.warning(f"The published case could not be loaded. {error}")
            return
        st.caption(CASES[case_key]["description"])
        st.caption("Held-out example from the [Brain Hemorrhage Segmentation Dataset](https://huggingface.co/datasets/Wendy-Fly/BHSD).")

    if values is not None:
        with st.container(key="ct-results-comparison"):
            cards = st.columns(3, gap="small")
            controls = st.container()
            with controls:
                slice_col, opacity_col = st.columns([2, 1])
                with slice_col:
                    count = values.shape[2]
                    initial = (int(reference.sum(axis=(0, 1)).argmax())
                               if reference is not None and reference.any() else count // 2)
                    key = active_id if case == "Current CT volume" else case
                    index = st.slider("Axial slice", 1, count, initial + 1, key=f"result-{key}") - 1 if count > 1 else 0
                opacity = opacity_col.slider("Overlay opacity", 0, 100, 65) / 100
            section = volume_section(values, 2, index)
            for column, (label, mask, color, placeholder) in zip(cards, (
                ("Original CT", None, "original", ""),
                ("Reference annotation · cyan", reference, "reference", "Supply an aligned reference mask."),
                ("Model prediction · orange", prediction, "prediction", "Generate an overlay in the Scan Viewer."),
            )):
                with column, st.container(border=True, key=f"ct-results-{color}"):
                    st.subheader(label)
                    if color != "original" and mask is None:
                        st.info(placeholder)
                        continue
                    overlay = volume_section(mask, 2, index) if mask is not None else None
                    st.image(render_slice(section, 40, 120, mask=overlay, opacity=opacity,
                                          warm=color == "prediction"), width="stretch")
                    if overlay is None:
                        st.caption(f"Unmasked input · Slice {index + 1} / {count}")
                    else:
                        st.caption(f"{np.count_nonzero(overlay):,} marked pixels on this slice")
            if reference is not None and prediction is not None:
                total = int(reference.sum() + prediction.sum())
                dice = f"{2 * np.count_nonzero(reference & prediction) / total:.3f}" if total else "Undefined"
                st.metric("Selected volume Dice", dice)


def volume_prediction():
    st.subheader("Predict on a CT volume")
    st.caption("Use a de-identified axial head CT in NIfTI format. Experimental research output.")
    selected_volume = st.session_state.get("analysis_volume")
    upload = None
    if selected_volume is None:
        upload = st.file_uploader("Research CT volume (.nii or .nii.gz)", type=["nii", "gz"])
    else:
        st.caption(f"Selected file: {selected_volume[0]}")
    project = Path(__file__).resolve().parents[1]
    example = project / "data" / "model-review" / "strong" / "scan.nii.gz"
    use_example = selected_volume is None and upload is None and st.toggle(
        "Use the BHSD research example", disabled=not example.is_file(),
    )
    checkpoint = project / "models" / "ct-segmenter-balanced-expanded.pt"
    if not checkpoint.is_file():
        checkpoint = project / "assets" / "checkpoints" / checkpoint.name
    if selected_volume is None and upload is None and not use_example:
        st.session_state.pop("volume_prediction", None)
        return
    contents = selected_volume[1] if selected_volume is not None else (
        example.read_bytes() if use_example else upload.getvalue()
    )
    fingerprint = hashlib.sha256(contents).hexdigest()
    image_panel, controls = st.columns([3, 1.15], gap="medium")
    with controls, st.container(border=True, key="ct-viewer-controls"):
        st.subheader("Viewer controls")
        try:
            import nibabel as nib
            from nibabel.filebasedimages import ImageFileError
            from nibabel.spatialimages import HeaderDataError
        except ImportError as error:
            st.error(f"Volume viewer setup is incomplete. {error}")
            return

        saved_id, scan_data = st.session_state.get("viewer_volume", (None, None))
        if saved_id != fingerprint:
            st.session_state.pop("viewer_volume", None)
            try:
                scan = nib.as_closest_canonical(decode_nifti(contents))
                if len(scan.shape) != 3 or min(scan.shape) < 1:
                    raise ValueError("Choose a non-empty 3D CT volume.")
                if nib.aff2axcodes(scan.affine) != ("R", "A", "S"):
                    raise ValueError("The volume could not be oriented for viewing.")
                values = scan.get_fdata(dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("The scan contains invalid pixel values.")
                spacing = tuple(float(step) for step in nib.affines.voxel_sizes(scan.affine))
            except (ValueError, OSError, EOFError, ImageFileError, HeaderDataError) as error:
                st.error(f"Unable to open the CT volume. {error}")
                return
            scan_data = (scan, values, spacing)
            st.session_state["viewer_volume"] = (fingerprint, scan_data)
        scan, values, spacing = scan_data

        plane = st.segmented_control("View plane", ["Axial", "Coronal", "Sagittal"],
                                     default="Axial") or "Axial"
        center, width = viewer_window("volume")
        ready = checkpoint.is_file()
        prediction_id = (fingerprint, checkpoint.stat().st_mtime_ns) if ready else None
        saved_id, prediction = st.session_state.get("volume_prediction", (None, None))
        if prediction_id is None or saved_id != prediction_id:
            prediction = None
            st.session_state.pop("volume_prediction", None)
        if st.button("Generate experimental overlay", type="primary", disabled=not ready):
            try:
                from predict_segmentation import predict_mask

                with st.spinner("Running segmentation on axial slices"):
                    prediction = predict_mask(scan, checkpoint).astype(bool)
                st.session_state["volume_prediction"] = (prediction_id, prediction)
            except (ImportError, ValueError, OSError, RuntimeError, KeyError) as error:
                st.error(f"Unable to generate an overlay. {error}")
        if not ready:
            st.caption("The segmentation checkpoint is unavailable.")
        if prediction is not None:
            show_overlay, opacity = viewer_overlay("Model overlay", "volume")
            if not prediction.any():
                st.caption("The model produced an empty mask.")
        else:
            show_overlay, opacity = False, 0.0

    axis = {"Axial": 2, "Coronal": 1, "Sagittal": 0}[plane]
    with image_panel, st.container(border=True, key="ct-viewer-stage"):
        st.caption(f"{plane} view · {values.shape[axis]} slices")
        preview = st.container()
        initial = values.shape[axis] // 2
        index = viewer_slice_index(plane, values.shape[axis], initial, fingerprint)
        section = volume_section(values, axis, index)
        overlay = volume_section(prediction, axis, index) if show_overlay else None
        rendered = render_slice(section, center, width, mask=overlay, opacity=opacity)
        height, columns = section.shape
        x_axis, y_axis = ((1, 2), (0, 2), (0, 1))[axis]
        ratio = columns * spacing[x_axis] / (height * spacing[y_axis]) if spacing[y_axis] > 0 else None
        preview.image(viewer_image(rendered, ratio), width="stretch")
        if overlay is not None:
            st.caption(f"Cyan shows the model output · {np.count_nonzero(overlay):,} marked pixels on this slice.")


def viewer_window(key):
    choices = {"Brain": (40, 80), "Subdural": (50, 150), "Bone": (500, 2000), "Custom": (40, 80)}
    preset = st.segmented_control("Window preset", list(choices), default="Brain",
                                  key=f"{key}-window") or "Brain"
    center, width = choices[preset]
    if preset == "Custom":
        center = st.slider("Window level (HU)", -200, 1000, center, key=f"{key}-level")
        width = st.slider("Window width (HU)", 1, 3000, width, key=f"{key}-width")
    else:
        st.caption(f"Width {width} HU · Level {center} HU")
    return center, width


def viewer_overlay(label, key):
    visible = st.toggle(label, value=True, key=f"{key}-overlay")
    opacity = st.slider("Overlay opacity", 0, 100, 65, disabled=not visible,
                        key=f"{key}-opacity") / 100
    return visible, opacity


def viewer_slice_index(plane, count, initial, fingerprint):
    if count == 1:
        return 0
    key = f"viewer-index-{fingerprint}-{plane}"
    current = st.session_state.setdefault(key, initial + 1)
    previous, step, following = st.columns([1, 5, 1])
    with previous:
        st.button("Previous", key=f"{key}-prev", disabled=current <= 1,
                  on_click=st.session_state.update, args=({key: current - 1},))
    with following:
        st.button("Next", key=f"{key}-next", disabled=current >= count,
                  on_click=st.session_state.update, args=({key: current + 1},))
    return step.slider(f"{plane} slice", 1, count, key=key) - 1


def volume_section(volume, axis, index):
    section = np.take(volume, index, axis=axis).T[::-1]
    return section[:, ::-1] if axis != 0 else section


def viewer_image(rendered, ratio=None):
    image = Image.fromarray(rendered)
    if ratio is not None and np.isfinite(ratio) and ratio > 0:
        width = max(1, round(768 * min(1, ratio)))
        height = max(1, round(768 * min(1, 1 / ratio)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def landing():
    from title_page import title_page

    title_page(upload_page)


def upload_scan():
    from title_page import workspace_entry

    workspace_entry()
    st.html(Path(__file__).with_name("upload_page.css"))
    st.html("""
        <div class="ct-upload-rail">
            <a href="./">CT HEMORRHAGE</a>
            <span>SCIENTIFIC PROTOTYPE</span>
        </div>
        <header class="ct-upload-intro">
            <h1>Upload a scan for analysis</h1>
            <p>Choose a CT volume or a single DICOM slice to open the scan viewer.</p>
        </header>
    """)
    with st.container(key="ct-upload-panel"):
        volume_col, dicom_col = st.columns(2, gap="large")
        with volume_col:
            st.html("""
                <div class="ct-upload-option">
                    <h2>CT volume</h2>
                    <p class="ct-upload-format">3D NIfTI · .nii or .nii.gz</p>
                    <p>Generate an experimental overlay for a full scan.</p>
                </div>
            """)
            volume = st.file_uploader("Upload a CT volume", type=["nii", "gz"], key="upload-volume", label_visibility="collapsed")
        with dicom_col:
            st.html("""
                <div class="ct-upload-option">
                    <h2>DICOM slice</h2>
                    <p class="ct-upload-format">Single CT image · .dcm</p>
                    <p>Run the classifier to view experimental subtype scores.</p>
                </div>
            """)
            dicom = st.file_uploader("Upload a DICOM slice", type=["dcm"], key="upload-dicom", label_visibility="collapsed")

    if volume is not None and not volume.name.lower().endswith((".nii", ".nii.gz")):
        st.error("Choose a NIfTI volume ending in .nii or .nii.gz.")
        volume = None
    if volume is not None:
        st.session_state.pop("viewer_volume", None)
        st.session_state.pop("analysis_scan", None)
        st.session_state.pop("volume_prediction", None)
        st.session_state["analysis_volume"] = (volume.name, volume.getvalue())
        st.session_state["analysis_source"] = "volume-prediction"
        st.switch_page(analysis_page)
    if dicom is not None:
        st.session_state.pop("analysis_volume", None)
        st.session_state.pop("volume_prediction", None)
        st.session_state["analysis_scan"] = (dicom.name, dicom.getvalue())
        st.session_state["analysis_source"] = "slice"
        st.switch_page(analysis_page)

    with st.container(key="ct-upload-teaching"):
        st.html("""
            <div class="ct-upload-teaching">
                <h2>No scan on hand?</h2>
                <p>View a published head CT teaching image.</p>
            </div>
        """)
        if st.button("Open teaching scan"):
            for key in ("analysis_scan", "analysis_volume", "volume_prediction"):
                st.session_state.pop(key, None)
            st.session_state["analysis_source"] = "slice"
            st.switch_page(analysis_page)
        if st.button("Open CT volume + reference mask"):
            st.session_state["analysis_source"] = "volume"
            st.switch_page(analysis_page)
        local_example = Path(__file__).resolve().parents[1] / "data/model-review/strong/scan.nii.gz"
        if local_example.is_file() and st.button("Open local BHSD volume"):
            st.session_state.pop("analysis_volume", None)
            st.session_state["analysis_source"] = "volume-prediction"
            st.switch_page(analysis_page)
    st.html("""
        <p class="ct-upload-foot">
            Use de-identified research scans only. The hosted app processes uploads on its server.
            Scientific prototype. Not for diagnostic purposes.
        </p>
    """)


def workspace_redirect():
    st.switch_page(upload_page)


def analysis():
    from title_page import workspace_entry

    view_route = st.session_state.get("analysis_source", "slice")

    workspace_entry()
    st.html(Path(__file__).with_name("scan_viewer.css"))
    st.html("""
        <div class="ct-viewer-rail">
            <span>CT HEMORRHAGE</span>
            <span>SCIENTIFIC PROTOTYPE</span>
        </div>
    """)
    st.title("Scan viewer")
    st.caption("Research use only. Not for diagnostic purposes.")
    if st.session_state.get("analysis_source") == "volume-prediction":
        volume_prediction()
        return
    if st.session_state.get("analysis_source") == "review":
        st.switch_page(results_page)
    if st.session_state.get("analysis_source") == "volume":
        annotated_case()
        return
    with st.expander("Replace DICOM slice or open a reference mask"):
        uploaded = st.file_uploader("Research CT slice (.dcm)", type=["dcm"], key="analysis-dicom")
        if st.button("Open CT volume and mask"):
            st.session_state["analysis_source"] = "volume"
            st.rerun()
    case = st.session_state.get("analysis_scan")
    if uploaded is not None:
        case = (uploaded.name, uploaded.getvalue())
        st.session_state["analysis_scan"] = case
    if case is None:
        image_col, details = st.columns([3, 1.15], gap="medium")
        with image_col, st.container(border=True, key="ct-viewer-stage"):
            st.caption("Axial view · Teaching image")
            st.image(TEACHING_CT, width=560)
        with details, st.container(border=True, key="ct-viewer-controls"):
            st.subheader("Teaching example")
            st.write("The authors identify a thalamic hemorrhage on this head CT.")
            st.caption("A published JPEG is available for visual review. Upload a CT volume or DICOM for research analysis.")
            st.markdown("[Yadav et al. (2007)](https://commons.wikimedia.org/wiki/File:Intracerebral_hemorrhage.jpg) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)")
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
    image_panel, metrics = st.columns([3, 1.15], gap="medium")
    with metrics, st.container(border=True, key="ct-viewer-controls"):
        st.subheader("Viewer controls")
        center, width = viewer_window("dicom")
        mask = None
        with st.expander("Add a reference mask"):
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
        show_mask, opacity = viewer_overlay("Reference mask", "dicom") if mask is not None else (False, 0.0)
        st.divider()
        st.subheader("Research classifier")
        st.caption("Uncalibrated scores for this slice.")
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
            st.metric("Any hemorrhage · raw score", f"{float(scores['any']):.3f}")
            st.markdown("**Subtype scores**")
            for label, value in scores.items():
                if label != "any":
                    st.progress(value, text=f"{label.capitalize()} · {value:.3f}")
    with image_panel, st.container(border=True, key="ct-viewer-stage"):
        st.caption(f"Single DICOM slice · {name}")
        image = render_slice(values, center, width, inverted, mask if show_mask else None, opacity)
        st.image(viewer_image(image), width="stretch")
        st.download_button("Export slice PNG", png_bytes(image), "ct-slice.png", "image/png")
        if mask is not None and show_mask:
            st.caption(f"Cyan shows the supplied mask · {np.count_nonzero(mask):,} marked pixels.")


def analysis_redirect():
    st.switch_page(analysis_page)


def anatomy():
    from anatomy_page import anatomy_page

    anatomy_page(results_page, upload_page)


def main():
    global landing_page, upload_page, analysis_page, results_page, anatomy_route
    st.set_page_config(page_title="CT Slice Lab", layout="wide")
    landing_page = st.Page(landing, title="CT Hemorrhage", default=True)
    upload_page = st.Page(upload_scan, title="Upload a scan", url_path="upload")
    legacy_workspace = st.Page(workspace_redirect, title="Research workspace", url_path="workspace")
    analysis_page = st.Page(analysis, title="Scan viewer", url_path="viewer")
    results_page = st.Page(model_review, title="Model results", url_path="results")
    anatomy_route = st.Page(anatomy, title="Interactive anatomy", url_path="anatomy")
    legacy_analysis = st.Page(analysis_redirect, title="Scan analysis", url_path="analysis")
    from methodology_page import methodology_page

    methodology = st.Page(methodology_page, title="Methodology", url_path="methodology")
    links = {
        "Home": landing_page, "Upload": upload_page, "Scan Viewer": analysis_page,
        "Model Results": results_page, "Anatomy": anatomy_route, "Methodology": methodology,
    }
    current = st.navigation([*links.values(), legacy_workspace, legacy_analysis], position="hidden")
    if current in (legacy_workspace, legacy_analysis):
        current.run()
        return

    st.html(Path(__file__).with_name("site_nav.css"))
    with st.container(key="ct-site-nav", horizontal=True, wrap=True, gap="xsmall"):
        for label, page in links.items():
            st.page_link(page, label=label)
    current.run()


def png_bytes(image):
    from matplotlib import image as mpl_image

    buffer = io.BytesIO()
    mpl_image.imsave(buffer, image, format="png")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
