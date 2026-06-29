"""Streamlit demo: the Agentic Financial Analyst over REAL SEC filings.

Framed for non-technical visitors (recruiters, HR). Ask a question about real public
companies; a hand-rolled ReAct agent looks up exact figures from the companies' filed XBRL
data, computes over them deterministically, and shows every step and source — or abstains.

Local:           streamlit run ledgerlens/ui/app.py   (reads .env)
Streamlit Cloud: set LEDGERLENS_* secrets in the dashboard (bridged to env below).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly
# (Streamlit Cloud puts the script's own folder on sys.path, not the repo root).
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st  # noqa: E402

from ledgerlens.agent import ResearchAgent, WorkspaceRegistry, make_dispatch  # noqa: E402
from ledgerlens.ingest.edgar import EdgarClient  # noqa: E402
from ledgerlens.llm import LLMClient  # noqa: E402

# Bridge Streamlit Cloud secrets -> environment so pydantic-settings picks them up.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - no secrets file locally; .env is used instead
    pass

_UA = os.environ.get("LEDGERLENS_SEC_USER_AGENT", "Agentic Financial Analyst portfolio demo")

_EXAMPLES = [
    "How fast did Apple's (AAPL) revenue grow in its most recent fiscal year?",
    "Compare Apple (AAPL) and Microsoft (MSFT) on revenue growth — which grew faster?",
    "What was NVIDIA's (NVDA) net income in its most recent fiscal year?",
]


@st.cache_resource
def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(EdgarClient(_UA))


st.set_page_config(page_title="Agentic Financial Analyst", page_icon="📊", layout="centered")

with st.sidebar:
    st.header("Why trust this?")
    st.markdown(
        "- **It never does the math itself.** A calculator computes every number; the AI only "
        "decides what to look up.\n"
        "- **Every figure is from the filing.** Numbers come straight from each company's "
        "official SEC XBRL data, with the source shown.\n"
        "- **It can say \"I'm not sure.\"** If a figure can't be verified, it abstains instead "
        "of guessing."
    )
    st.caption("Live data from SEC EDGAR. Model: Azure gpt-4.1-mini.")

st.title("📊 Agentic Financial Analyst")
st.subheader("Ask about real companies' financials — answers you can actually trust.")
st.markdown(
    "Analysts spend hours pulling numbers out of 100–300 page filings. The obvious shortcut — "
    "*paste it into ChatGPT* — fails three ways:"
)
col1, col2, col3 = st.columns(3)
col1.error("**Too big**\n\nThe filing doesn't fit in the model's context window.")
col2.error("**Made-up numbers**\n\nThe model hallucinates figures that look right.")
col3.error("**Bad arithmetic**\n\nEven correct numbers get miscalculated.")
st.markdown(
    "**This agent fixes all three.** It reads real SEC filings, looks up each figure from the "
    "company's official data, computes with a real calculator, and **shows its work and its "
    "sources** — step by step."
)

st.divider()
st.markdown("### Ask the analyst")
choice = st.selectbox("Pick an example (or edit it below)", _EXAMPLES)
task = st.text_area("Question", value=choice, height=80)

if st.button("Research it", type="primary"):
    with st.spinner("Reading filings, looking up figures, computing… (first lookup can take ~30s)"):
        try:
            agent = ResearchAgent(LLMClient(), make_dispatch(_registry()), max_steps=8)
            report = agent.run(task)
        except Exception as exc:  # noqa: BLE001 - surface any provider/network error to the user
            st.error(f"Something went wrong: {exc}")
            st.stop()

    st.divider()
    st.success(f"### {report.summary}")
    if report.trend and report.trend.lower() != "n/a":
        st.caption(f"Verdict: **{report.trend}**")

    st.markdown(
        "**How the analyst worked it out** — every figure sourced, every calc verified:"
    )
    for step in report.steps:
        icon = "✅" if step.ok else "⚠️"
        st.markdown(f"{icon} **{step.tool}** · {step.target}")
        if step.thought:
            st.caption(step.thought)
        st.code(step.observation, language="text")

    tel = report.telemetry
    st.caption(
        f"{tel.tool_calls} tool calls · {tel.succeeded} verified · "
        f"{tel.corrections} self-correction(s)"
    )
