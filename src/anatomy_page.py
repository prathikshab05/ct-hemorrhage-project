"""Educational anatomy and imaging vocabulary."""

from pathlib import Path

import streamlit as st

from dashboard import brain_image_uri
from title_page import workspace_entry


REGIONS = {
    "Frontal lobe": (24, 36, "Front of the cerebrum", "Helps with planning, decision-making, and voluntary movement."),
    "Parietal lobe": (59, 22, "Upper rear part of the cerebrum", "Processes touch and helps you keep track of where your body is in space."),
    "Temporal lobe": (46, 63, "Lower side of the cerebrum", "Supports hearing, memory, and understanding language."),
    "Occipital lobe": (82, 47, "Back of the cerebrum", "Processes visual information."),
    "Cerebellum": (70, 76, "Below the back of the cerebrum", "Helps coordinate movements and maintain balance."),
}

GLOSSARY = {
    "CT and scan planes": (
        ("Axial", "A horizontal slice through the head."),
        ("Coronal", "A slice that separates the front and back."),
        ("Sagittal", "A slice that separates left and right."),
        ("Hounsfield unit (HU)", "A value on the CT scale that measures how strongly tissue attenuates X-rays relative to water."),
        ("Window width and level", "Width sets the range of CT values mapped to shades of gray. Level sets the center of that range."),
    ),
    "Hemorrhage labels": (
        ("Intracranial hemorrhage", "Bleeding anywhere inside the skull."),
        ("Epidural", "Bleeding between the skull and the dura, the outermost protective membrane."),
        ("Subdural", "Bleeding between the dura and the arachnoid membrane."),
        ("Subarachnoid", "Bleeding into the fluid-filled space between the arachnoid and pia membranes."),
        ("Intraparenchymal", "Bleeding within brain tissue."),
        ("Intraventricular", "Bleeding into the brain’s fluid-filled ventricles."),
    ),
    "Model and evaluation": (
        ("Classification", "The DICOM classifier returns experimental scores for whether a slice contains hemorrhage and for each subtype."),
        ("Segmentation", "The volume model marks individual pixels on axial slices. The marks form a predicted mask."),
        ("Reference annotation", "The labeled region used to compare against the model’s mask."),
        ("Voxel", "A pixel’s three-dimensional counterpart in a CT volume."),
        ("Dice score", "How much two masks overlap, from 0 for no overlap to 1 for identical nonempty masks."),
    ),
}


def anatomy_page(results_page, upload_page):
    workspace_entry()
    st.html(Path(__file__).with_name("scan_viewer.css"))
    st.html(Path(__file__).with_name("anatomy.css"))
    st.html("""
        <div class="ct-viewer-rail">
            <span>CT HEMORRHAGE</span>
            <span>EDUCATIONAL REFERENCE</span>
        </div>
    """)
    back, upload = st.columns(2)
    back.page_link(results_page, label="Back to model results", icon=":material/arrow_back:")
    upload.page_link(upload_page, label="Upload a scan", icon=":material/upload:")
    st.title("Interactive anatomy")

    with st.container(key="anatomy-explorer"):
        selected = st.segmented_control(
            "Explore a region", list(REGIONS), default="Frontal lobe", key="anatomy-region",
        ) or "Frontal lobe"
        art, details = st.columns([2.6, 1], gap="medium")
        markers = "".join(
            f'<div class="ct-anatomy-pin{" is-selected" if name == selected else ""}" '
            f'style="left:{x}%;top:{y}%"><span>{name}</span></div>'
            for name, (x, y, _, _) in REGIONS.items()
        )
        art.html(f"""
            <figure class="ct-anatomy-figure">
                <div class="ct-anatomy-image">
                    <img src="{brain_image_uri()}" alt="Neon side view of the brain">
                    <div aria-hidden="true">{markers}</div>
                </div>
                <figcaption>Educational illustration · Approximate locations · Not a patient scan</figcaption>
            </figure>
        """)
        _, _, location, function = REGIONS[selected]
        with details, st.container(border=True, key="anatomy-detail"):
            st.subheader(selected)
            st.caption(location)
            st.write(function)
            st.markdown("[Brain anatomy reference · NINDS](https://www.ninds.nih.gov/sites/default/files/2025-05/know-your-brain-brian-basics.pdf)")

    st.header("CT and model glossary")
    for tab, (title, entries) in zip(st.tabs(list(GLOSSARY)), GLOSSARY.items()):
        with tab:
            terms = "".join(f"<dt>{term}</dt><dd>{definition}</dd>" for term, definition in entries)
            st.html(f'<dl class="ct-anatomy-glossary">{terms}</dl>')
            if title == "Hemorrhage labels":
                st.markdown("[Hemorrhage terminology · NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK470242/)")
