"""Research methods, data provenance, and evaluation."""

from pathlib import Path

import streamlit as st

from title_page import workspace_entry


REPOSITORY = "https://github.com/prathikshab05/ct-hemorrhage-project"


def methodology_page():
    workspace_entry()
    st.html(Path(__file__).with_name("scan_viewer.css"))
    st.html(Path(__file__).with_name("methodology.css"))
    st.html("""
        <div class="ct-viewer-rail">
            <span>CT HEMORRHAGE</span>
            <span>SCIENTIFIC PROTOTYPE</span>
        </div>
    """)
    heading, link = st.columns([3, 1], vertical_alignment="center")
    heading.title("Methodology")
    link.link_button("View source on GitHub", REPOSITORY, icon=":material/code:")
    st.caption("Research prototype. Not for diagnostic purposes.")

    with st.container(key="ct-method-grid"):
        sources, training, limits = st.columns(3, gap="medium")
        with sources, st.container(border=True):
            st.subheader("Data sources")
            st.markdown(
                "**BHSD** supplied CT volumes with voxel-level annotations for the segmentation experiment. "
                "**RSNA** supplied labeled DICOM slices for a separate classifier."
            )
            with st.expander("Dataset links and image credits"):
                st.markdown(
                    "- [Brain Hemorrhage Segmentation Dataset](https://huggingface.co/datasets/Wendy-Fly/BHSD)\n"
                    "- [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)\n"
                    "- [Seg-CQ500 figure used on the landing page](https://doi.org/10.3389/fnimg.2023.1157565)"
                )
        with training, st.container(border=True):
            st.subheader("Training and validation")
            st.markdown(
                "**2D U-Net** · three adjacent axial slices → center-slice mask. "
                "The expanded run used 59 volumes for training, 15 for threshold selection, "
                "and two holdouts of 8 and 12 volumes."
            )
            with st.expander("Training and split details"):
                st.write(
                    "The model trained for eight epochs with AdamW, horizontal flips, and a combined weighted binary cross-entropy "
                    "and Dice loss. Hemorrhage subtypes share one foreground label. Archive identifiers were used "
                    "for grouping because distinct patient identities could not be verified."
                )
        with limits, st.container(border=True):
            st.subheader("Current limitations")
            st.markdown(
                "Overlap varied substantially across both test sets. All contained annotated hemorrhage, "
                "so scan-level specificity was unmeasured."
            )
            with st.expander("What the evaluation can tell us"):
                st.write(
                    "The evaluation gives a preliminary measure of mask overlap on this held-out subset. "
                    "It does not establish performance on bleed-negative scans, across hospitals, or for individual "
                    "hemorrhage subtypes. The classifier’s scores are uncalibrated."
                )

    show_segmentation_benchmarks("ct-method-eval")
    with st.expander("Find the training and evaluation code"):
        st.markdown(
            f"- [Segmentation trainer]({REPOSITORY}/blob/main/src/train_segmentation.py)\n"
            f"- [Native-grid evaluator]({REPOSITORY}/blob/main/src/evaluate_segmentation.py)\n"
            f"- [Reproduction steps and experiment notes]({REPOSITORY}#segmentation-evaluation)"
        )


def show_segmentation_benchmarks(key):
    with st.container(border=True, key=key):
        st.subheader("Held-out segmentation results")
        dice, precision, recall = st.columns(3)
        dice.metric("Mean Dice", "0.495")
        precision.metric("Mean voxel precision", "67.1%")
        recall.metric("Mean voxel recall", "45.3%")
        st.caption(
            "Original holdout · 8 BHSD volumes. The previous model scored 0.354 mean Dice."
        )
        st.caption("Fresh holdout · 12 BHSD volumes. Mean Dice rose from 0.267 to 0.351. "
                   "Both cohorts contained only scans with annotated hemorrhage.")
