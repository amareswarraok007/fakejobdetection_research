"""CareerVerify-XAI — Analyze page (PRD §11). Run:  streamlit run app/main.py

This file only draws the page. Every calculation is done by ``cvx.service``.
The model store folder can be changed with the environment variable CVX_MODEL_STORE.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from style import APP_CSS, hero_html, rule_card_html, verdict_html  # noqa: E402

from cvx import service  # noqa: E402
from cvx.artifacts.load import ModelStoreError  # noqa: E402
from cvx.evaluate.plots import shap_bar_figure  # noqa: E402
from cvx.schema import InputError  # noqa: E402

STORE_PATH = os.environ.get("CVX_MODEL_STORE", "app/model_store")
YES_NO = ("No", "Yes")
HERO_PILLS = ("100% Offline & Local", "SHA-256 Verified Models", "SHAP + LIME + Rules R1–R7")


@st.cache_resource
def get_store(path: str) -> Any:
    """Load and check the models once per app process (hash, versions, smoke scores)."""
    return service.open_store(path)


def sidebar(store: Any) -> None:
    """Brand header, verification badge, risk legend and expandable system check."""
    info = service.health(store)
    models = ", ".join(info.get("models", []))
    st.sidebar.markdown(
        '<div class="cvx-sidebar-brand">'
        "<strong>CareerVerify-XAI</strong>"
        "<span>Leakage-Aware Job Fraud Detector</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div class="cvx-status-ok">Models verified · {models}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Runs locally on your laptop. Nothing you paste is saved or sent online.")
    with st.sidebar.expander("How overall risk is computed", expanded=False):
        st.markdown(
            "- **Model level:** Low (`< t_r90`), Medium (`t_r90` to `t_f1`), High (`>= t_f1`).\n"
            "- **Rules level:** Low (none / Low rules), Medium (1 Medium), "
            "High (any High or 2+ Medium).\n"
            "- **Overall risk:** the **higher** of the two (scores are never added)."
        )
    with st.sidebar.expander("System check", expanded=False):
        st.json(info)


def full_posting_fields(options: dict[str, list[str]]) -> dict[str, Any]:
    """Inputs for a structured job posting (inside the form)."""
    st.markdown('<div class="cvx-section-label">Job Details</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 1.5, 1])
    form: dict[str, Any] = {
        "title": col_a.text_input("Job title *", placeholder="e.g., Junior Python Developer"),
        "company_name": col_b.text_input("Company name", placeholder="e.g., Acme Cloud Pvt Ltd"),
        "salary_range": col_c.text_input("Salary range", placeholder="e.g., 50000-70000"),
    }
    form["description"] = st.text_area(
        "Job description *",
        height=160,
        placeholder="Paste the main role summary and responsibilities…",
    )
    t1, t2, t3 = st.columns(3)
    form["company_profile"] = t1.text_area("Company description", height=100)
    form["requirements"] = t2.text_area("Requirements", height=100)
    form["benefits"] = t3.text_area("Benefits", height=100)
    st.markdown(
        '<div class="cvx-section-label">Role Attributes &amp; Signals</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    for column, key in zip(
        columns,
        ("employment_type", "required_experience", "required_education", "function"),
        strict=True,
    ):
        choices = options.get(key, ["Unknown"])
        form[key] = column.selectbox(
            key.replace("_", " ").capitalize(),
            choices,
            index=choices.index("Unknown") if "Unknown" in choices else 0,
        )
    flags = st.columns(3)
    for column, key, label in zip(
        flags,
        ("has_company_logo", "has_questions", "telecommuting"),
        ("Company logo shown?", "Screening questions?", "Remote job?"),
        strict=True,
    ):
        form[key] = column.radio(label, YES_NO, horizontal=True) == "Yes"
    return form


def input_form(store: Any, mode: str) -> tuple[dict[str, Any], str, bool]:
    """The analysis form; returns (form values, model choice, submitted?)."""
    with st.form("analyze"):
        if mode == "full":
            form = full_posting_fields(service.category_options(store))
        else:
            st.markdown('<div class="cvx-section-label">Job Message</div>', unsafe_allow_html=True)
            form = {
                "message": st.text_area(
                    "Paste the job message *",
                    height=200,
                    placeholder="Paste a job offer from WhatsApp, Telegram, email, or a portal…",
                )
            }
        st.markdown(
            '<div class="cvx-section-label">Optional Recruiter Verification (Rules R3 & R4)</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([2, 2, 1.2])
        form["recruiter_email"] = c1.text_input(
            "Recruiter email (optional, rules only)", placeholder="e.g., hr@company.com"
        )
        form["company_website"] = c2.text_input(
            "Company website (optional, rules only)", placeholder="e.g., https://company.com"
        )
        choice = c3.selectbox("Model", service.available_choices(store, mode))
        submitted = st.form_submit_button("Analyze", type="primary", width="stretch")
    return form, choice, submitted


def show_rules(result: dict[str, Any]) -> None:
    """The separate 'Safety checks (rules, not AI)' card."""
    with st.container(border=True):
        st.markdown("#### Safety checks (rules, not AI)")
        if not result["rules"]:
            st.success("No safety rule fired for this posting.")
            return
        for hit in result["rules"]:
            st.markdown(rule_card_html(hit), unsafe_allow_html=True)


def show_shap_block(result: dict[str, Any]) -> None:
    """Top-10 SHAP chart alongside its matching accessible data table."""
    with st.container(border=True):
        st.markdown("#### Feature Attribution (Top-10 SHAP)")
        chart_col, table_col = st.columns([1.45, 1])
        items = [(row["feature"], row["shap"]) for row in result["shap"]]
        chart_col.pyplot(shap_bar_figure(items, "Top-10 SHAP features"))
        frame = pd.DataFrame(result["shap"])
        if not frame.empty:
            label = lambda v: "Towards fake (+)" if v > 0 else "Towards genuine (−)"  # noqa: E731
            frame = frame.assign(direction=frame["shap"].map(label))
        table_col.dataframe(frame, hide_index=True, width="stretch")


def show_result(result: dict[str, Any]) -> None:
    """Verdict banner, KPI metrics, reasons, highlighted text, safety rules and SHAP chart."""
    risk = result["risk"]
    st.subheader(f"Overall risk: {risk['overall']}")
    st.markdown(verdict_html(result), unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Model score (probability of fake)",
        f"{result['score']:.3f}",
        help=f"Model: {result['model_id']}; thresholds: {result['thresholds']}",
    )
    k2.metric("Model risk band", risk["model"])
    k3.metric("Safety rules band", risk["rules"])
    left, right = st.columns([1.25, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Why this score? (Top reasons)")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
            st.info(result["caution"])
        with st.container(border=True):
            st.markdown("#### Highlighted text")
            st.markdown(
                '<div class="cvx-legend">'
                '<span><mark style="background:rgba(220,38,38,0.35);color:#111;border-radius:3px">'
                "Red +</mark> pushes towards fake</span>"
                '<span><mark style="background:rgba(22,163,74,0.35);color:#111;border-radius:3px">'
                "Green −</mark> pushes towards genuine</span>"
                "<span>Hover any word for its exact SHAP score</span></div>"
                f'<div class="cvx-highlight-box">{result["highlight_html"]}</div>',
                unsafe_allow_html=True,
            )
    with right:
        show_rules(result)
    show_shap_block(result)


def lime_section(store: Any) -> None:
    """'Explain with LIME' button (outside the form) and the SHAP comparison."""
    form, mode, result = (st.session_state[k] for k in ("form", "mode", "result"))
    with st.container(border=True):
        st.markdown("#### Cross-check with LIME (Optional)")
        st.caption(
            "Runs a 1,000-sample perturbation explainer on the same posting to check how closely "
            "LIME agrees with SHAP."
        )
        if st.button("Explain with LIME"):
            with st.spinner("Running LIME (about 1,000 model calls)…"):
                st.session_state["lime"] = service.lime_compare(
                    form, mode, result["model_id"], result["words"], store
                )
        lime = st.session_state.get("lime")
        if lime:
            m1, m2 = st.columns(2)
            m1.metric("Top-5 overlap with SHAP (Jaccard)", f"{lime['jaccard_top5']:.2f}")
            m2.metric("LIME runtime", f"{lime['seconds']:.2f} s")
            st.dataframe(pd.DataFrame(lime["lime"]), hide_index=True, width="stretch")


def _mode_label(mode: str) -> str:
    """Readable label for each input mode."""
    return (
        "Full posting (text + metadata)"
        if mode == "full"
        else "Text only (WhatsApp / email message)"
    )


def main() -> None:
    """Draw the Analyze page."""
    st.set_page_config(page_title="CareerVerify-XAI", page_icon="🔎", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        hero_html(
            "CareerVerify-XAI",
            "Leakage-aware fake job posting detector with verified SHAP & LIME explanations "
            "and deterministic safety rules.",
            HERO_PILLS,
        ),
        unsafe_allow_html=True,
    )
    try:
        store = get_store(STORE_PATH)
    except ModelStoreError as error:
        st.error(f"The app cannot start: {error}")
        st.stop()
    sidebar(store)
    mode = st.radio("Input mode", ("full", "text"), horizontal=True, format_func=_mode_label)
    form, choice, submitted = input_form(store, mode)
    if submitted:
        try:
            st.session_state.update(
                result=service.analyze(form, mode, choice, store), form=form, mode=mode, lime=None
            )
        except InputError as error:
            st.error(str(error))
    if st.session_state.get("result"):
        show_result(st.session_state["result"])
        lime_section(store)


main()
