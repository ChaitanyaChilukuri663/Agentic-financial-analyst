"""Streamlit demo for the Agentic Financial Analyst (LedgerLens).

Framed for non-technical visitors (recruiters, HR): it states the real problem, explains the
approach in plain language, and for every answer shows the computed steps, the sources, and
whether the system trusts the result — or abstains.

Local:           streamlit run ledgerlens/ui/app.py   (reads .env)
Streamlit Cloud: set LEDGERLENS_* secrets in the dashboard (bridged to env below).
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly
# (Streamlit Cloud puts the script's own folder on sys.path, not the repo root).
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st  # noqa: E402

from ledgerlens.engine import answer_question  # noqa: E402
from ledgerlens.llm import LLMClient  # noqa: E402

# Bridge Streamlit Cloud secrets -> environment so pydantic-settings picks them up.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - no secrets file locally; .env is used instead
    pass

_EXAMPLES = {
    "Change between two years": (
        "What was the change in net revenue from 2022 to 2023, in millions?",
        "Net revenue was $5,829 million in 2023 and $5,735 million in 2022. "
        "Cost of sales was $3,100 million in 2023.",
    ),
    "Add two line items": (
        "What were total residential mortgages for 2013 and 2012, in millions?",
        "Residential mortgages were 1,356 in 2013 and 2,220 in 2012.",
    ),
    "Share of a total (ratio)": (
        "What share of total purchase commitments is due after 2024?",
        "Purchase commitments due after 2024 were 25,048; total commitments were 44,572.",
    ),
}


def _pretty(answer: str | None) -> str:
    if not answer:
        return ""
    try:
        return f"{Decimal(answer):.4g}"
    except (InvalidOperation, ValueError):
        return answer


st.set_page_config(page_title="Agentic Financial Analyst", page_icon="📊", layout="centered")

with st.sidebar:
    st.header("Why trust this?")
    st.markdown(
        "- **It never does the math itself.** The AI writes a *plan*; a plain calculator "
        "computes every number.\n"
        "- **Every number is sourced.** Each figure is traced back to the document.\n"
        "- **It can say \"I'm not sure.\"** If a figure can't be verified, it abstains instead "
        "of guessing."
    )
    st.caption("Evaluated on FinQA + real SEC 10-K filings. Model: Azure gpt-4.1-mini.")

st.title("📊 Agentic Financial Analyst")
st.subheader("Answers from financial filings you can actually trust.")
st.markdown(
    "Analysts spend hours pulling numbers out of 100–300 page filings and computing ratios by "
    "hand. The obvious shortcut — *paste it into ChatGPT* — fails three ways:"
)
col1, col2, col3 = st.columns(3)
col1.error("**Too big**\n\nThe filing doesn't fit in the model's context window.")
col2.error("**Made-up numbers**\n\nThe model hallucinates figures that look right.")
col3.error("**Bad arithmetic**\n\nEven correct numbers get miscalculated.")
st.markdown(
    "**This tool fixes all three.** It finds the relevant evidence, has the AI propose a "
    "*reasoning plan* (not the answer), runs the math with a real calculator, and **shows its "
    "work and its sources** — or declines when it can't verify the result."
)

st.divider()
st.markdown("### Try it")
choice = st.selectbox("Pick an example (or edit the evidence below)", list(_EXAMPLES))
default_question, default_context = _EXAMPLES[choice]
question = st.text_input("Question", value=default_question)
context = st.text_area(
    "Evidence (what the analyst is allowed to use)",
    value=default_context,
    height=140,
    help="Tip: delete a number here and re-run to watch the system refuse to guess.",
)

if st.button("Compute the answer", type="primary"):
    with st.spinner("Finding evidence, planning, and computing…"):
        try:
            result = answer_question(LLMClient(), question, context)
        except Exception as exc:  # noqa: BLE001 - surface any provider/config error to the user
            st.error(f"Something went wrong: {exc}")
            st.stop()

    st.divider()
    if result.answered:
        st.success(f"## ✅ {_pretty(result.answer)}")
        st.caption("Verified — every number below was computed and traced to the evidence.")
    else:
        st.warning("## ⚠️ The system declined to answer")
        st.caption("It couldn't verify the result, so it abstained instead of guessing:")
        st.markdown("\n".join(f"- {reason}" for reason in result.abstain_reasons))

    if result.steps:
        st.markdown("**How it was computed** — by a calculator, not the AI:")
        st.table(
            [{"step": s.op, "inputs": ", ".join(s.args), "result": s.result} for s in result.steps]
        )
    if result.operands:
        st.markdown("**Where each number comes from:**")
        st.table(
            [{"number": o.value, "meaning": o.name, "source": o.citation} for o in result.operands]
        )
    if result.program:
        with st.expander("Technical detail — the reasoning program"):
            st.code(result.program, language="text")
            st.caption(
                "The AI emits this symbolic program; a deterministic engine executes it, so the "
                "arithmetic cannot be hallucinated."
            )
