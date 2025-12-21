# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path

DELPHI_LOGO = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║          ██████╗ ███████╗██╗     ██████╗ ██╗  ██╗██╗                     ║
║          ██╔══██╗██╔════╝██║     ██╔══██╗██║  ██║██║                     ║
║          ██║  ██║█████╗  ██║     ██████╔╝███████║██║                     ║
║          ██║  ██║██╔══╝  ██║     ██╔═══╝ ██╔══██║██║                     ║
║          ██████╔╝███████╗███████╗██║     ██║  ██║██║                     ║
║          ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝                     ║
║                                                                          ║
║                    🏛️  TIME SERIES ORACLE  🔮                           ║
║              "Γνῶθι σεαυτόν" - Connais-toi toi-même                      ║
║                        ... et tes données                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

ORACLE_TEMPLE = r"""
                            ⚡
                           /|\
                          / | \
                    _____|_____|_____
                   |  ═══════════  |
                   |   🏛️ ORACLE   |
                   |  ═══════════  |
                   |_______________|
                   |▓|▓|▓|▓|▓|▓|▓|▓|
                   |▓|▓|▓|▓|▓|▓|▓|▓|
               ____|▓|▓|▓|▓|▓|▓|▓|▓|____
              |  ═════════════════════  |
              |    Temple de Delphes    |
              |  ═════════════════════  |
              |_________________________|
"""

PREDICTIONS_BANNER = r"""
╔═══════════════════════════════════════╗
║     🔮  PRÉDICTIONS DE L'ORACLE  🔮   ║
╚═══════════════════════════════════════╝
"""

BOXJENKINS_BANNER = r"""
╔════════════════════════════════════════╗
║   📦  MÉTHODOLOGIE BOX-JENKINS  📊    ║
║                                        ║
║   1️⃣  Identification                   ║
║   2️⃣  Spécification                    ║
║   3️⃣  Estimation                       ║
║   4️⃣  Diagnostic                       ║
║   5️⃣  Prévision                        ║
╚════════════════════════════════════════╝
"""

DEEPLEARNING_BANNER = r"""
╔════════════════════════════════════════╗
║   🧠  DEEP LEARNING NEURAL ORACLE  ⚡  ║
║                                        ║
║   • LSTM - Long Short-Term Memory     ║
║   • GRU  - Gated Recurrent Unit       ║
║   • VAR  - Vector Autoregression      ║
╚════════════════════════════════════════╝
"""

FOOTER = r"""
═══════════════════════════════════════════════════════════════
        Développé avec 🏛️ inspiration grecque antique
                  et 🧠 science moderne
═══════════════════════════════════════════════════════════════
"""

DELPHI_IMAGES = {
    "temple": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/The_Tholos_at_the_sanctuary_of_Athena_Pronaia_%283471168708%29.jpg/1280px-The_Tholos_at_the_sanctuary_of_Athena_Pronaia_%283471168708%29.jpg",
    "ruins": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Delphi_Temple_of_Apollo.jpg/1280px-Delphi_Temple_of_Apollo.jpg",
    "theater": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Delphi_-_Ancient_Theater.jpg/1280px-Delphi_-_Ancient_Theater.jpg",
    "oracle_illustration": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/John_Collier_-_Priestess_of_Delphi.jpg/800px-John_Collier_-_Priestess_of_Delphi.jpg",
}



def load_css(file_name: str = "delphi_style.css") -> None:
    css_path = Path(__file__).parent / file_name
    if not css_path.exists():
        st.warning(f"CSS introuvable: {css_path}")
        return

    # Essaye plusieurs encodages (Windows + UTF-8 BOM)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            css = css_path.read_text(encoding=enc)
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            return
        except UnicodeDecodeError:
            continue

    st.error("❌ Impossible de lire le CSS: encodage non supporté. Ré-enregistre le fichier en UTF-8.")

def apply_branding(show_logo: bool = True, show_temple: bool = False) -> None:
    """Applique CSS + éléments visuels (ASCII art, images, footer)."""
    load_css("delphi_style.css")

    if show_logo:
        st.markdown(f"```text\n{DELPHI_LOGO}\n```")

    # Sidebar: un petit clin d'œil
    st.sidebar.markdown("### 🏛️ DELPHI")
    st.sidebar.caption("Time Series Oracle 🔮")

    # Option : image décorative
    with st.sidebar.expander("📸 Delphi (images)"):
        st.image(DELPHI_IMAGES["temple"], caption="Tholos at Delphi (Wikimedia)", use_container_width=True)
        st.image(DELPHI_IMAGES["oracle_illustration"], caption="Priestess of Delphi (Wikimedia)", use_container_width=True)

    if show_temple:
        st.markdown(f"```text\n{ORACLE_TEMPLE}\n```")

def banner_boxjenkins():
    st.markdown(f"```text\n{BOXJENKINS_BANNER}\n```")

def banner_deeplearning():
    st.markdown(f"```text\n{DEEPLEARNING_BANNER}\n```")

def banner_predictions():
    st.markdown(f"```text\n{PREDICTIONS_BANNER}\n```")

def footer():
    st.markdown(f"<div class='footer'><pre>{FOOTER}</pre></div>", unsafe_allow_html=True)
