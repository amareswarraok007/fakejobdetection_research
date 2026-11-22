"""About page (PRD §11.1): what the tool can and cannot do."""

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = str(Path(__file__).resolve().parents[1])
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from style import APP_CSS, hero_html  # noqa: E402

from cvx.explain.reasons import CAUTION  # noqa: E402

st.set_page_config(page_title="About · CareerVerify-XAI", page_icon="🔎", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(
    hero_html(
        "About CareerVerify-XAI",
        "How the detector works, how safety rules complement the ML model, "
        "and key limitations to keep in mind.",
        ("EMSCAD Benchmark", "Faithfulness-Tested XAI", "Zero Data Retention"),
    ),
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("#### What it does")
        st.markdown(
            "Scores a structured job posting or a pasted job message using a leakage-aware ML "
            "model trained on EMSCAD, explains every prediction with **SHAP** (and optional "
            "**LIME**), and runs separate deterministic safety rules (**R1–R7**)."
        )
with c2:
    with st.container(border=True):
        st.markdown("#### Safety rules (R1–R7)")
        st.markdown(
            "- **High:** Fee / deposit / UPI requests (R1); Aadhaar / PAN / OTP / bank (R2).\n"
            "- **Medium:** Free webmail (R3), domain mismatch (R4), WhatsApp / Telegram (R5), "
            "guaranteed job or unrealistic pay (R6).\n"
            "- **Low:** Urgency pressure (R7)."
        )
with c3:
    with st.container(border=True):
        st.markdown("#### Privacy & integrity")
        st.markdown(
            "Everything runs **100% locally** on your laptop. Nothing you type is stored on disk "
            "or transmitted over the internet. Models load only after passing **SHA-256 hash**, "
            "library-version, and **smoke-test** startup checks."
        )

with st.container(border=True):
    st.markdown("#### What it cannot do (Limitations)")
    st.markdown(
        "- **Not proof of fraud:** A high score means the posting resembles fake postings in "
        "historical data; always verify the employer independently.\n"
        "- **Dataset age:** EMSCAD postings come from 2012–2014 and mostly outside India. "
        "Modern WhatsApp, Telegram, and UPI scams look different, so the separate safety rules "
        "matter as much as the ML score.\n"
        "- **Email/domain rules:** Rules R3 and R4 could not be evaluated on EMSCAD because "
        "the dataset has no email or website columns.\n"
        "- **Model vs. truth:** SHAP and LIME explain what the model learned, not absolute truth."
    )

st.warning(CAUTION)
