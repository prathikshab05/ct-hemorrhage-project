"""Landing page for the CT research workspace."""

from base64 import b64encode
from pathlib import Path

import streamlit as st


def title_page(workspace_page):
    asset = Path(__file__).resolve().parents[1] / "assets" / "ct-title-reference.webp"
    image = b64encode(asset.read_bytes()).decode("ascii")
    css = Path(__file__).with_name("title_page.css").read_text(encoding="utf-8")
    st.html(f"""
        <style>{css}</style>
        <div class="ct-title-scene">
          <header class="ct-title-rail">
            <span><i></i> CT SLICE LAB</span>
            <span>SCIENTIFIC PROTOTYPE</span>
          </header>
          <div class="ct-title-copy">
            <h1><span>CT</span> HEMORRHAGE</h1>
            <p>Head CT imaging research</p>
          </div>
          <figure class="ct-title-figure">
            <div class="ct-title-grid">
              <div class="ct-title-art">
                <img src="data:image/webp;base64,{image}"
                     alt="Lateral illustration of a brain with orange highlights">
              </div>
              <span class="ct-title-label ct-title-label-left">Lateral brain illustration</span>
              <span class="ct-title-label ct-title-label-right">Illustrative highlights</span>
            </div>
          </figure>
          <div class="ct-title-actions">
            <a class="ct-title-enter" href="./{workspace_page.url_path}">
              <span>Access research prototype</span><span aria-hidden="true">→</span>
            </a>
            <p>Scientific prototype. Not for diagnostic purposes.</p>
          </div>
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
