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
            <a class="ct-title-down" href="#prototype-purpose">
              Why this prototype exists <span aria-hidden="true">&darr;</span>
            </a>
          </div>
          </div>
          <section class="ct-title-purpose" id="prototype-purpose" tabindex="-1"
                   aria-labelledby="prototype-purpose-heading">
            <header>
              <h2 id="prototype-purpose-heading">Why this prototype exists</h2>
              <p>CT Slice Lab helps students and medical imaging researchers explore how AI outlines bleeding on head CT and see where its predictions differ from reference annotations.</p>
            </header>
            <div class="ct-purpose-grid">
              <article>
                <h3>Which scan gets reviewed first?</h3>
                <p>Suspected bleeding inside the skull can need urgent review. FDA-cleared tools such as Aidoc BriefCase flag suspected bleeds for earlier review by a radiologist. <a href="https://www.accessdata.fda.gov/cdrh_docs/pdf18/K180647.pdf">FDA clearance summary</a></p>
              </article>
              <article>
                <h3>When the radiologist is off-site</h3>
                <p>Some rural hospitals rely on remote radiologists. Flagging suspected urgent findings is relevant to those reading queues too. <a href="https://car.ca/wp-content/uploads/2025/10/CAR_Position_Statement_on_Remote_Reporting_EN_2025-09.pdf">Canadian Association of Radiologists</a></p>
              </article>
              <article>
                <h3>Small bleeds can be hard to see</h3>
                <p>A thin subdural bleed can blend into the edge of the skull on CT. Subtle findings are an important test of any proposed second reader. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5307932/">Imaging review</a></p>
              </article>
              <article>
                <h3>Where is it, and has it grown?</h3>
                <p>Segmentation outlines a region across slices. With known voxel sizes, those outlines can help estimate bleed volume and track changes between scans. <a href="https://doi.org/10.1161/STR.0000000000000407">AHA/ASA guideline</a></p>
              </article>
            </div>
            <div class="ct-purpose-next">
              <div>
                <h3>Try the teaching scan</h3>
                <p>Open the built-in example to compare the predicted outline with the reference, or upload a de-identified research scan. This prototype is for education and research only.</p>
              </div>
              <a class="ct-title-enter" href="./{workspace_page.url_path}">
                Open scan workspace <span aria-hidden="true">&rarr;</span>
              </a>
            </div>
          </section>
          <section class="ct-title-evidence" aria-label="Prototype research details">
            <article>
              <h2>Dataset</h2>
              <p class="ct-title-fact">BHSD + RSNA</p>
              <p><a href="https://huggingface.co/datasets/Wendy-Fly/BHSD">BHSD</a> supplied 94 annotated volumes for the segmentation experiment.
                <a href="https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection">RSNA</a> supplied labeled slices for the separate classifier.</p>
            </article>
            <article>
              <h2>Model architecture</h2>
              <p class="ct-title-fact">2D U-Net with adjacent slices</p>
              <p>The segmentation model uses three neighboring axial slices to predict a binary mask for the center slice.</p>
            </article>
            <article>
              <h2>Held-out evaluation</h2>
              <p class="ct-title-fact">0.495 mean Dice</p>
              <p>Mean mask overlap rose from 0.354 to 0.495 on the original eight BHSD test scans and from 0.267 to 0.351 on 12 additional test scans.</p>
            </article>
          </section>
        </div>
        <script>
          (() => {{
            const scene = document.querySelector(".ct-title-scene");
            const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
            const modified = (event) => event.button || event.metaKey || event.ctrlKey ||
                                        event.shiftKey || event.altKey;
            const purpose = scene?.querySelector(".ct-title-purpose");
            scene?.querySelector(".ct-title-down")?.addEventListener("click", (event) => {{
              if (!purpose || modified(event)) return;
              event.preventDefault();
              purpose.focus({{preventScroll: true}});
              purpose.scrollIntoView({{behavior: reducedMotion.matches ? "instant" : "smooth"}});
              if (!reducedMotion.matches) {{
                purpose.querySelector("header").animate(
                  [{{opacity: .5, transform: "translateY(12px)"}},
                   {{opacity: 1, transform: "translateY(0)"}}],
                  {{duration: 460, easing: "ease-out"}});
              }}
            }});
            scene?.querySelectorAll(".ct-title-enter").forEach((link) => link.addEventListener("click", (event) => {{
              if (modified(event) || reducedMotion.matches) return;
              event.preventDefault();
              if (scene.classList.contains("ct-title-leaving")) return;
              scene.classList.add("ct-title-leaving");
              window.setTimeout(() => window.location.assign(link.href), 340);
            }}));
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
