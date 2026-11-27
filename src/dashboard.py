"""Presentation components for the CT research dashboard."""

import streamlit as st


from base64 import b64encode
from html import escape
from pathlib import Path


REGIONS = {
    "Frontal lobe": (26, 36, "Toward the front of the brain."),
    "Parietal lobe": (61, 23, "Along the upper rear part of the cortex."),
    "Temporal lobe": (45, 59, "Along the lower side of the cortex."),
    "Occipital lobe": (78, 48, "Toward the back of the brain."),
    "Cerebellum": (70, 76, "At the lower back, beneath the cortex."),
}


def dashboard_header():
    st.markdown("""
    <style>
      .stApp {
        background: radial-gradient(ellipse at 50% 10%, #0a2940, transparent 54%),
          repeating-linear-gradient(0deg, transparent 0 35px, #16304055 35px 36px),
          repeating-linear-gradient(90deg, transparent 0 35px, #16304055 35px 36px), #060e17;
      }
      .block-container {max-width: 1540px; padding-top: 3.5rem}
      .stApp, .stApp button, .stApp input {
        font-family: "Segoe UI Variable", "Segoe UI", system-ui, sans-serif;
      }
      .stApp h1, .stApp h2, .stApp h3 {font-weight: 650; letter-spacing: -.035em}
      [data-testid="stVerticalBlockBorderWrapper"] {background: #0b1a26ee; border-radius: 18px}
      .ct-heading {padding: 12px 0 19px; border-bottom: 1px solid #234455; margin-bottom: 18px}
      .ct-heading h1 {font-size: clamp(2rem, 4vw, 3.5rem); letter-spacing: -.045em; padding: 0}
      .ct-heading span {color: #68e8ff}
      .ct-heading p {color: #a9bdcb; margin: 4px 0 0}
      .ct-brain {background: radial-gradient(ellipse, #14344b, #07131f 68%);
                 border: 1px solid #1c3b50; border-radius: 16px; padding: 14px}
      .ct-stage {position: relative; overflow: hidden; border-radius: 10px}
      .ct-stage img {display: block; width: 100%; height: auto}
      .ct-point {position: absolute; width: 13px; height: 13px; border-radius: 50%;
        background: #ffb988; border: 2px solid #ffe5d1; transform: translate(-50%,-50%);
        box-shadow: 0 0 0 8px #ff8d5d48, 0 0 27px 12px #ff6948a0;
        animation: ct-marker .45s ease-out both}
      .ct-point span {position: absolute; bottom: 17px; left: 50%; transform: translateX(-50%);
        color: #effaff; background: #061522ee; border: 1px solid #74dcf7;
        border-radius: 5px; padding: 4px 8px; white-space: nowrap; font-size: .78rem}
      @keyframes ct-marker {from {opacity: 0; scale: .6} to {opacity: 1; scale: 1}}
      @media (prefers-reduced-motion: reduce) {.ct-point {animation: none}}
      .ct-brain p {margin: 0; color: #9ab7c8; font-size: .82rem}
    </style>
    <div class="ct-heading"><h1>CT <span>Slice Lab</span></h1>
    <p>Head CT imaging workbench · Research prototype</p></div>
    """, unsafe_allow_html=True)


@st.cache_data
def brain_image_uri():
    asset = Path(__file__).resolve().parents[1] / "assets" / "neon-brain.webp"
    return "data:image/webp;base64," + b64encode(asset.read_bytes()).decode("ascii")


def brain_reference():
    st.subheader("Brain anatomy")
    labels = st.toggle("Show anatomy labels", value=True)
    if labels:
        area = st.radio("Explore a region", REGIONS.keys(), horizontal=True)
        x, y, description = REGIONS[area]
        marker = (
            f'<div class="ct-point" style="left:{x}%;top:{y}%">'
            f'<span>{area}</span></div>'
        )
    else:
        marker = ""
    st.markdown(
        f'<div class="ct-brain"><div class="ct-stage">'
        f'<img src="{brain_image_uri()}" alt="Neon lateral illustration of a human brain">'
        f'{marker}</div></div>',
        unsafe_allow_html=True,
    )
    if labels:
        st.caption(description)
    st.caption("Illustrative anatomy. Labels are approximate and independent of the CT.")


def scan_workspace_style():
    st.markdown("""
    <style>
      .ct-workspace {padding: 12px 0 18px; border-bottom: 1px solid #29474b; margin-bottom: 12px}
      .ct-workspace strong {font-size: clamp(1.45rem,3vw,2.55rem); color: #91f0e7; letter-spacing: -.04em}
      .ct-workspace p {margin: 4px 0 0; color: #9eb7b9}
      .ct-montage {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px;
        border:1px solid #345456;border-radius:12px;overflow:hidden;background:#345456}
      .ct-montage.single {grid-template-columns:minmax(0,1fr)}
      .ct-slice {position:relative;background:#020607;display:grid;place-items:center;
        min-width:0;aspect-ratio:1.19;overflow:hidden;margin:0}
      .ct-slice img {width:100%;height:100%;object-fit:contain;position:absolute;inset:0}
      .ct-slice figcaption {position:absolute;top:12px;left:12px;padding:5px 9px;border-radius:5px;
        background:#142327ee;color:#eef8f8;font:650 .85rem "Segoe UI",system-ui,sans-serif;letter-spacing:.055em}
      .ct-slice.selected {box-shadow:inset 0 0 0 2px #73f1df}
      .ct-slice.selected:after {content:"";position:absolute;inset:0;border:2px solid #73f1df;pointer-events:none}
      .ct-slice.selected figcaption {color:#95faed}
      .ct-montage.single .ct-slice {aspect-ratio:1.55;max-height:560px}
      .ct-rail-title {font-size:.77rem;font-weight:700;color:#91f0e7;letter-spacing:.12em;margin-bottom:8px}
      @media(max-width:620px) {.ct-montage {grid-template-columns:1fr}}
    </style>
    <div class="ct-workspace"><strong>CT Slices &amp; Analysis</strong>
    <p>Research imaging workspace</p></div>
    """, unsafe_allow_html=True)


def scan_montage(tiles):
    """Display PNG tiles with overlaid view labels."""
    cards = []
    for png, label, selected in tiles:
        encoded = b64encode(png).decode("ascii")
        safe_label = escape(label)
        selected_class = " selected" if selected else ""
        cards.append(
            f'<figure class="ct-slice{selected_class}">'
            f'<img src="data:image/png;base64,{encoded}" alt="{safe_label}">'
            f'<figcaption>{safe_label}</figcaption></figure>'
        )
    layout = " single" if len(cards) == 1 else ""
    st.markdown(f'<div class="ct-montage{layout}">{"".join(cards)}</div>', unsafe_allow_html=True)
