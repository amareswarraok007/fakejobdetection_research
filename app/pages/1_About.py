"""About page (PRD §11.1): what the tool can and cannot do."""

import streamlit as st

from cvx.explain.reasons import CAUTION

st.title("About CareerVerify-XAI")
st.markdown(
    """
**What it does.** It scores a job posting or a pasted job message with a machine-learning
model trained on the public EMSCAD dataset, explains the score with SHAP (and, on request,
LIME), and runs separate safety rules for common scam signals such as payment requests.

**What it cannot do.**
- It cannot prove that a posting is fake or genuine. A high score means the posting looks
  like fake postings the model has seen, nothing more.
- EMSCAD postings are from 2012–2014 and mostly from outside India. Today's WhatsApp,
  Telegram and UPI scams may look different, so the safety rules matter as much as the score.
- Rules R3 and R4 (recruiter email checks) could not be tested on EMSCAD, which has no email
  or website columns.
- The explanations describe what the model learned, not the truth about a company.

**Privacy.** Everything runs on this computer. Nothing you type is stored or sent anywhere.
"""
)
st.warning(CAUTION)
