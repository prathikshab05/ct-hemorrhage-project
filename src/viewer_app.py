"""Research CT viewer and scan-specific model results."""

import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
from pydicom.pixels import apply_modality_lut
from PIL import Image, ImageDraw, ImageFilter

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


def model_review():
    from methodology_page import show_segmentation_benchmarks
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
    st.caption("Results for the current upload · Research use only. Not for diagnostic purposes.")
    source, name, fingerprint = st.session_state["active_scan"]
    st.subheader("Current upload")
    st.text(name)
    if source == "slice":
        score_id, scores = st.session_state.get("analysis_scores", (None, None))
        if score_id is None or score_id[0] != fingerprint or scores is None:
            st.info("Open the Scan Viewer to run the classifier for this slice.")
        else:
            show_classifier_scores(scores)
    else:
        render_current_volume(fingerprint)

    show_segmentation_benchmarks("ct-results-metrics")


def render_current_volume(active_id):
    from nibabel.filebasedimages import ImageFileError
    from nibabel.wrapstruct import WrapStructError

    scan, values, _ = st.session_state["viewer_volume"][1]
    saved_id, saved_mask = st.session_state.get("volume_prediction", (None, None))
    prediction = saved_mask if saved_id is not None and saved_id[0] == active_id else None
    reference = None
    uploaded = st.file_uploader("Optional aligned NIfTI reference mask", type=["nii", "gz"],
                                key=f"results-mask-{active_id}")
    if uploaded is not None:
        try:
            _, reference = aligned_volumes(scan, decode_nifti(uploaded.getvalue()))
        except (ValueError, OSError, EOFError, ImageFileError, WrapStructError) as error:
            st.error(f"Unable to open the reference mask. {error}")

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
                    index = st.slider("Axial slice", 1, count, initial + 1, key=f"result-{active_id}") - 1 if count > 1 else 0
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
            if prediction is not None:
                predicted_count = int(np.count_nonzero(prediction))
                model_count, match = st.columns(2)
                model_count.metric("Model-marked voxels · full volume", f"{predicted_count:,}")
                if reference is not None:
                    total = int(reference.sum()) + predicted_count
                    dice = f"{2 * np.count_nonzero(reference & prediction) / total:.3f}" if total else "Undefined"
                    match.metric("Dice · current volume", dice)
                else:
                    match.caption("Add an aligned reference mask to calculate Dice for this upload.")


def volume_prediction():
    st.subheader("Predict on a CT volume")
    st.caption("Use a de-identified axial head CT in NIfTI format. Experimental research output.")
    name, contents = st.session_state["analysis_volume"]
    st.text(name)
    project = Path(__file__).resolve().parents[1]
    checkpoint = project / "models" / "ct-segmenter-balanced-expanded.pt"
    if not checkpoint.is_file():
        checkpoint = project / "assets" / "checkpoints" / checkpoint.name

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
        st.session_state["active_scan"] = ("volume-prediction", name, fingerprint)

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
        output_metrics = st.container()
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
        predicted_slice = volume_section(prediction, axis, index) if prediction is not None else None
        overlay = predicted_slice if show_overlay else None
        rendered = render_slice(section, center, width, mask=overlay, opacity=opacity)
        height, columns = section.shape
        x_axis, y_axis = ((1, 2), (0, 2), (0, 1))[axis]
        ratio = columns * spacing[x_axis] / (height * spacing[y_axis]) if spacing[y_axis] > 0 else None
        preview.image(viewer_image(rendered, ratio), width="stretch")
        if overlay is not None:
            st.caption("Cyan shows the model output.")
    if predicted_slice is not None:
        with output_metrics:
            st.metric("Model-marked pixels · selected slice", f"{np.count_nonzero(predicted_slice):,}")
            st.metric("Model-marked voxels · full volume", f"{np.count_nonzero(prediction):,}")


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

    if volume is not None or dicom is not None:
        for key in ("active_scan", "viewer_volume", "analysis_scan", "analysis_volume",
                    "analysis_scores", "classifier_error", "volume_prediction"):
            st.session_state.pop(key, None)
        if volume is not None:
            if not volume.name.lower().endswith((".nii", ".nii.gz")):
                st.error("Choose a NIfTI volume ending in .nii or .nii.gz.")
            else:
                st.session_state["analysis_volume"] = (volume.name, volume.getvalue())
                st.session_state["analysis_source"] = "volume-prediction"
                st.switch_page(analysis_page)
        else:
            st.session_state["analysis_scan"] = (dicom.name, dicom.getvalue())
            st.session_state["analysis_source"] = "slice"
            st.switch_page(analysis_page)
    if "active_scan" not in st.session_state:
        st.caption("Upload a scan to unlock the Scan Viewer and Model Results.")

    with st.container(key="ct-upload-teaching"):
        st.subheader("No scan on hand?")
        st.caption("Explore a published CT example without uploading a file.")
        with st.expander("Open teaching scan"):
            asset = Path(__file__).resolve().parents[1] / "assets" / "ct-seg-cq500-preview.webp"
            st.image(str(asset), caption="Axial, coronal, and sagittal views of one Seg-CQ500 case.")
            st.caption(
                "Salmon shows the reference hemorrhage annotation. Blue shows the cavity outline "
                "from the source paper’s U-Net. This is a static teaching example."
            )
            st.markdown(
                "[Spahr et al. (2023), Fig. 1B](https://doi.org/10.3389/fnimg.2023.1157565) · "
                "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
            )

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
    case = st.session_state["analysis_scan"]
    name, contents = case
    try:
        values, inverted = read_slice(contents)
        if not np.isfinite(values).all():
            raise ValueError("The slice contains invalid pixel values.")
    except (ValueError, OSError, RuntimeError, NotImplementedError, AttributeError, pydicom.errors.InvalidDicomError) as error:
        st.error(f"Unable to open this slice. {error}")
        return

    model_dir = Path(__file__).resolve().parents[1] / "models"
    candidates = (
        model_dir / "ct-expanded-spatial.pt",
        model_dir.parent / "assets" / "checkpoints" / "ct-expanded-spatial.pt",
        model_dir / "ct-spatial-cnn.pt",
    )
    checkpoint = next((path for path in candidates if path.is_file()), candidates[-1])
    fingerprint = hashlib.sha256(contents).hexdigest()
    scan_id = (fingerprint, checkpoint.name)
    st.session_state["active_scan"] = ("slice", name, fingerprint)
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
        failed_id, failure = st.session_state.get("classifier_error", (None, None))
        if failed_id != scan_id:
            failure = None
        retry = st.button("Retry research classifier") if failure is not None else False
        if scores is None and (failure is None or retry):
            try:
                from research_model import load_model, score_slice

                with st.spinner("Running the research classifier"):
                    scores = score_slice(load_model(checkpoint), contents)
                st.session_state["analysis_scores"] = (scan_id, scores)
                st.session_state.pop("classifier_error", None)
                failure = None
            except (ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
                failure = str(error)
                st.session_state["classifier_error"] = (scan_id, failure)
                st.rerun()
        if failure is not None:
            st.error(f"Classifier unavailable. {failure}")
        if scores is not None:
            show_classifier_scores(scores)
    with image_panel, st.container(border=True, key="ct-viewer-stage"):
        st.caption(f"Single DICOM slice · {name}")
        image = render_slice(values, center, width, inverted, mask if show_mask else None, opacity)
        st.image(viewer_image(image), width="stretch")
        st.download_button("Export slice PNG", png_bytes(image), "ct-slice.png", "image/png")
        if mask is not None and show_mask:
            st.caption(f"Cyan shows the supplied mask · {np.count_nonzero(mask):,} marked pixels.")


def show_classifier_scores(scores):
    st.metric("Any hemorrhage · raw score", f"{float(scores['any']):.3f}")
    st.markdown("**Subtype scores**")
    for label, value in scores.items():
        if label != "any":
            st.progress(value, text=f"{label.capitalize()} · {value:.3f}")


def analysis_redirect():
    st.switch_page(analysis_page)


def anatomy():
    from anatomy_page import anatomy_page

    anatomy_page(upload_page)


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

    has_upload = any(key in st.session_state for key in ("analysis_scan", "analysis_volume"))
    if current == analysis_page and not has_upload:
        st.switch_page(upload_page)
    if current == results_page and "active_scan" not in st.session_state:
        st.switch_page(upload_page)

    st.html(Path(__file__).with_name("site_nav.css"))
    nav = st.container(key="ct-site-nav", horizontal=True, wrap=True, gap="xsmall")
    current.run()
    with nav:
        for label, page in links.items():
            if page in (analysis_page, results_page) and "active_scan" not in st.session_state:
                continue
            st.page_link(page, label=label)


def png_bytes(image):
    from matplotlib import image as mpl_image

    buffer = io.BytesIO()
    mpl_image.imsave(buffer, image, format="png")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
