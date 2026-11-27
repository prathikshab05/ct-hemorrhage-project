"""Presentation components for the CT research dashboard."""

import streamlit as st


from base64 import b64encode
from html import escape
from pathlib import Path


@st.cache_data
def brain_image_uri():
    asset = Path(__file__).resolve().parents[1] / "assets" / "neon-brain.webp"
    return "data:image/webp;base64," + b64encode(asset.read_bytes()).decode("ascii")


def scan_workspace_style(show_heading=True):
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
    """, unsafe_allow_html=True)
    if show_heading:
        st.html("""
          <div class="ct-workspace"><strong>CT Slices &amp; Analysis</strong>
          <p>Research imaging workspace</p></div>
        """)



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
