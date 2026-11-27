"""Landing page for the CT research workspace."""

from base64 import b64encode
from pathlib import Path

import streamlit as st


def title_page(workspace_page):
    asset = Path(__file__).resolve().parents[1] / "assets" / "ct-seg-cq500-preview.webp"
    image = b64encode(asset.read_bytes()).decode("ascii")
    css = Path(__file__).with_name("title_page.css").read_text(encoding="utf-8")
    st.html(f"""
        <style>{css}</style>
        <div class="ct-title-scene">
          <div class="ct-title-splash">
          <header class="ct-title-rail">
            <span><i></i> CT SLICE LAB</span>
            <span>SCIENTIFIC PROTOTYPE</span>
          </header>
          <div class="ct-title-copy">
            <h1><span>CT</span> HEMORRHAGE</h1>
            <p>Head CT imaging research</p>
          </div>
          <figure class="ct-title-figure">
            <div class="ct-title-views" style="--ct-preview: url(data:image/webp;base64,{image})">
              <div class="ct-title-view">
                <span>Axial</span>
                <div class="ct-title-scan" role="img" aria-label="Axial head CT with a reference hemorrhage mask"></div>
              </div>
              <div class="ct-title-view">
                <span>Coronal</span>
                <div class="ct-title-scan" role="img" aria-label="Coronal head CT with a reference hemorrhage mask"></div>
              </div>
              <div class="ct-title-view">
                <span>Sagittal</span>
                <div class="ct-title-scan" role="img" aria-label="Sagittal head CT with a reference hemorrhage mask"></div>
              </div>
            </div>
            <figcaption>
              <span class="ct-title-legend">
                <span><i class="ct-mask-key"></i>Reference hemorrhage mask</span>
                <span><i class="ct-cavity-key"></i>Intracranial cavity outline (source U-Net)</span>
              </span>
              <span>Published Seg-CQ500 example · Cropped from
                <a href="https://doi.org/10.3389/fnimg.2023.1157565">Spahr et al. (2023), Fig. 1B</a> ·
                <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
              </span>
            </figcaption>
          </figure>
          <div class="ct-title-actions">
            <a class="ct-title-enter" href="./{workspace_page.url_path}">
              <span>Access research prototype</span><span aria-hidden="true">→</span>
            </a>
            <p>Scientific prototype. Not for diagnostic purposes.</p>
          </div>
          </div>
          <section class="ct-title-evidence" aria-label="Prototype research details">
            <article>
              <h2>Dataset</h2>
              <p class="ct-title-fact">BHSD + RSNA</p>
              <p><a href="https://huggingface.co/datasets/Wendy-Fly/BHSD">BHSD</a> supplied 40 annotated volumes for the segmentation experiment.
                <a href="https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection">RSNA</a> supplied labeled slices for the separate classifier.</p>
            </article>
            <article>
              <h2>Model architecture</h2>
              <p class="ct-title-fact">2D convolutional network</p>
              <p>The segmentation model combines fine-detail and pooled features to produce one binary mask per axial slice.</p>
            </article>
            <article>
              <h2>Validation metrics</h2>
              <p class="ct-title-fact">0.354 mean Dice</p>
              <p>Mask overlap on eight held-out BHSD scans, all with annotated hemorrhage. Performance on bleed-negative scans is unmeasured.</p>
            </article>
          </section>
        </div>
        <script>
          (() => {{
            const scene = document.querySelector(".ct-title-scene");
            const link = scene?.querySelector(".ct-title-enter");
            link?.addEventListener("click", (event) => {{
              if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey ||
                  window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
              event.preventDefault();
              if (scene.classList.contains("ct-title-leaving")) return;
              scene.classList.add("ct-title-leaving");
              window.setTimeout(() => window.location.assign(link.href), 340);
            }});
            window.addEventListener("pageshow", () => scene?.classList.remove("ct-title-leaving"),
                                    {{once: true}});
          }})();
        </script>
    """, unsafe_allow_javascript=True)


def workspace_entry():
    st.html("""
      <style>
        @media (prefers-reduced-motion: no-preference) {
          .stApp .block-container { animation: ct-workspace-in .46s cubic-bezier(.16,1,.3,1) both; }
        }
        @keyframes ct-workspace-in {
          from { opacity: 0; transform: translate3d(0,24px,0); }
          to { opacity: 1; transform: translate3d(0,0,0); }
        }
      </style>
    """)
