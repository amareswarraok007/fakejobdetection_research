"""CareerVerify-XAI — Analyze page (PRD §11). Run:  streamlit run app/main.py

This file only draws the page. Every calculation is done by ``cvx.service``.
The model store folder can be changed with the environment variable CVX_MODEL_STORE.
"""

import os
from typing import Any

import pandas as pd
import streamlit as st

from cvx import service
from cvx.artifacts.load import ModelStoreError
from cvx.evaluate.plots import shap_bar_figure
from cvx.schema import InputError

STORE_PATH = os.environ.get("CVX_MODEL_STORE", "app/model_store")
YES_NO = ("No", "Yes")


@st.cache_resource
def get_store(path: str) -> Any:
    """Load and check the models once per app process (hash, versions, smoke scores)."""
    return service.open_store(path)


def sidebar(store: Any) -> None:
    """System check box shown on every run of the page."""
    with st.sidebar.expander("System check", expanded=False):
        st.json(service.health(store))


def full_posting_fields(options: dict[str, list[str]]) -> dict[str, Any]:
    """Inputs for a full posting (inside the form)."""
    form: dict[str, Any] = {
        "title": st.text_input("Job title *"),
        "company_name": st.text_input("Company name"),
        "company_profile": st.text_area("Company description"),
        "description": st.text_area("Job description *", height=180),
        "requirements": st.text_area("Requirements"),
        "benefits": st.text_area("Benefits"),
        "salary_range": st.text_input("Salary range"),
    }
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
            form = {"message": st.text_area("Paste the job message *", height=220)}
        form["recruiter_email"] = st.text_input("Recruiter email (optional, rules only)")
        form["company_website"] = st.text_input("Company website (optional, rules only)")
        choice = st.selectbox("Model", service.CHOICES)
        submitted = st.form_submit_button("Analyze")
    return form, choice, submitted


def show_rules(result: dict[str, Any]) -> None:
    """The separate 'Safety checks (rules, not AI)' box."""
    with st.container(border=True):
        st.subheader("Safety checks (rules, not AI)")
        if not result["rules"]:
            st.write("No rule fired.")
        for hit in result["rules"]:
            matched = ", ".join(f"‘{m}’" for m in hit["matched"])
            st.write(
                f"**{hit['rule_id']} · {hit['severity']}** — {hit['message']} (matched: {matched})"
            )


def show_result(result: dict[str, Any]) -> None:
    """Score, risk level, reasons, highlighted text, SHAP chart + table, rules."""
    risk = result["risk"]
    st.subheader(f"Overall risk: {risk['overall']}")
    st.write(f"Model: {risk['model']} · Safety rules: {risk['rules']} · Overall: {risk['overall']}")
    st.metric(
        "Model score (probability of fake)",
        f"{result['score']:.3f}",
        help=f"Model: {result['model_id']}; thresholds: {result['thresholds']}",
    )
    st.markdown("**Why (top reasons)**")
    for reason in result["reasons"]:
        st.write(f"- {reason}")
    st.info(result["caution"])
    st.markdown("**Highlighted text** (red + = towards fake, green − = towards genuine)")
    st.markdown(result["highlight_html"], unsafe_allow_html=True)  # built from escaped text
    items = [(row["feature"], row["shap"]) for row in result["shap"]]
    st.pyplot(shap_bar_figure(items, "Top-10 SHAP features"))
    st.dataframe(pd.DataFrame(result["shap"]), hide_index=True)  # same numbers as the chart
    show_rules(result)


def lime_section(store: Any) -> None:
    """'Explain with LIME' button (outside the form) and the SHAP comparison."""
    form, mode, result = (st.session_state[k] for k in ("form", "mode", "result"))
    if st.button("Explain with LIME"):
        with st.spinner("Running LIME (about 1,000 model calls)…"):
            st.session_state["lime"] = service.lime_compare(
                form, mode, result["model_id"], result["words"], store
            )
    lime = st.session_state.get("lime")
    if lime:
        st.write(
            f"Top-5 overlap with SHAP (Jaccard): {lime['jaccard_top5']:.2f} "
            f"· {lime['seconds']:.1f} s"
        )
        st.dataframe(pd.DataFrame(lime["lime"]), hide_index=True)


def main() -> None:
    """Draw the Analyze page."""
    st.set_page_config(page_title="CareerVerify-XAI", page_icon="🔎", layout="wide")
    st.title("CareerVerify-XAI: is this job posting fake?")
    try:
        store = get_store(STORE_PATH)
    except ModelStoreError as error:
        st.error(f"The app cannot start: {error}")
        st.stop()
    sidebar(store)
    mode = st.radio(
        "Input",
        ("full", "text"),
        horizontal=True,
        format_func=lambda m: "Full posting" if m == "full" else "Text only (message)",
    )
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
