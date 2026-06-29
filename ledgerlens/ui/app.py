"""Streamlit demo for LedgerLens.

Ask a question over financial evidence and see the LLM-proposed program, the
deterministic computation, the cited operands, and the gate verdict.

Local:           streamlit run ledgerlens/ui/app.py   (reads .env)
Streamlit Cloud: set LEDGERLENS_* secrets in the dashboard (bridged to env below).
"""

from __future__ import annotations

import os

import streamlit as st

from ledgerlens.engine import answer_question
from ledgerlens.llm import LLMClient

# Bridge Streamlit Cloud secrets -> environment so pydantic-settings picks them up.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - no secrets file locally; .env is used instead
    pass

_EXAMPLES = {
    "Percent change (revenue)": (
        "what was the percent change in net revenue from 2022 to 2023?",
        "Net revenue was $5,829 million in 2023 and $5,735 million in 2022. "
        "Cost of sales was $3,100 million in 2023.",
    ),
    "Sum of two line items": (
        "what were total residential mortgages for 2013 and 2012, in millions?",
        "Residential mortgages were 1,356 in 2013 and 2,220 in 2012.",
    ),
    "Ratio as a percentage": (
        "what percentage of total purchase commitments are due after 2024?",
        "Purchase commitments due after 2024 were 25,048; total commitments were 44,572.",
    ),
}

st.set_page_config(page_title="LedgerLens", page_icon="📊", layout="centered")
st.title("📊 LedgerLens")
st.caption(
    "Grounded financial-QA: the LLM proposes a reasoning *program*; a deterministic engine "
    "computes it. Every answer is cited and verified — or the system abstains."
)

choice = st.selectbox("Example", list(_EXAMPLES))
default_question, default_context = _EXAMPLES[choice]
question = st.text_input("Question", value=default_question)
context = st.text_area("Evidence", value=default_context, height=160)

if st.button("Compute", type="primary"):
    with st.spinner("Proposing a program and computing…"):
        try:
            result = answer_question(LLMClient(), question, context)
        except Exception as exc:  # noqa: BLE001 - surface any provider/config error to the user
            st.error(f"Failed: {exc}")
            st.stop()

    if result.answered:
        st.success(f"**Answer: {result.answer}**  ·  verified ✓")
    else:
        st.warning("**Abstained** — the answer did not pass the trust gates.")
        st.write("Reasons: " + ", ".join(result.abstain_reasons))

    if result.program:
        st.subheader("Reasoning program")
        st.code(result.program, language="text")

    if result.steps:
        st.subheader("Computed steps")
        st.table(
            [{"op": s.op, "args": ", ".join(s.args), "result": s.result} for s in result.steps]
        )

    if result.operands:
        st.subheader("Operands & citations")
        st.table(
            [{"operand": o.name, "value": o.value, "citation": o.citation} for o in result.operands]
        )
