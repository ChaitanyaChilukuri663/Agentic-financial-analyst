"""Streamlit demo for the Agentic Financial Analyst.

Four tabs, framed for non-technical visitors (recruiters, HR):
  Overview      — the problem, the flagship features, headline metrics.
  Try it        — ask the agent about real companies (with LLM-suggested questions).
  Results       — the evaluation numbers as charts.
  How it's built — the architecture diagram + the stack.

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

import pandas as pd  # noqa: E402 - bundled with streamlit
import streamlit as st  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from ledgerlens.agent import ResearchAgent, WorkspaceRegistry, make_dispatch  # noqa: E402
from ledgerlens.ingest.edgar import EdgarClient  # noqa: E402
from ledgerlens.llm import LLMClient  # noqa: E402

# Bridge Streamlit Cloud secrets -> environment so pydantic-settings picks them up.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001 - no secrets file locally; .env is used instead
    pass

_UA = os.environ.get(
    "LEDGERLENS_SEC_USER_AGENT", "Agentic Financial Analyst demo (contact@example.com)"
)
_REPO = "https://github.com/ChaitanyaChilukuri663/Agentic-financial-analyst"
_EXAMPLES = [
    "How fast did Apple's (AAPL) revenue grow in its most recent fiscal year?",
    "Compare Apple (AAPL) and Microsoft (MSFT) on revenue growth — which grew faster?",
    "What was NVIDIA's (NVDA) net income in its most recent fiscal year?",
]
_SUGGEST_PROMPT = (
    "Suggest 3 short, specific questions an analyst could ask about a large public company's "
    "most recent annual financials (e.g. revenue growth, margin, net income), using real "
    "tickers like AAPL, MSFT, or NVDA."
)


class _Suggestions(BaseModel):
    """A few suggested questions for the demo."""

    questions: list[str] = Field(
        description="3 short, specific questions about a company's financials."
    )


@st.cache_resource
def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(EdgarClient(_UA))


def _set_question(value: str) -> None:
    st.session_state.question = value


def _overview() -> None:
    st.subheader("Answers from financial filings you can actually trust.")
    st.markdown(
        "The language model **never produces a number** — it proposes a *plan*, and a "
        "deterministic engine does the math. Every figure is pulled from the filing, cited, and "
        "verified — or the system abstains."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Executor accuracy", "99.5%", "8,281 expert programs")
    m2.metric("Retrieval recall@5", "85.6%", "hybrid beats BM25/dense")
    m3.metric("False-abstentions", "0%", "gates buy precision")
    m4.metric("Data", "Real 10-Ks", "live from SEC EDGAR")

    st.divider()
    st.markdown("#### Why not just paste the filing into ChatGPT?")
    c1, c2, c3 = st.columns(3)
    c1.error("**Too big** — a 10-K doesn't fit the context window.")
    c2.error("**Made-up numbers** — it hallucinates figures.")
    c3.error("**Bad arithmetic** — it miscalculates.")

    st.divider()
    st.markdown("#### Flagship features")
    f1, f2 = st.columns(2)
    f1.markdown(
        "**🧮 No hallucinated math**\n\nThe LLM emits a program like "
        "`subtract(5829, 5735), divide(#0, 5735)`; a calculator runs it."
    )
    f1.markdown("**🔗 Every number cited**\n\nEach figure traces to the filing or filed XBRL data.")
    f2.markdown("**🛑 Knows when to abstain**\n\nIf a figure can't be verified, it refuses.")
    f2.markdown(
        "**🤖 Agentic**\n\nA self-correcting agent that works across companies — every figure "
        "still verified."
    )
    st.info("Open **🔎 Try it** to ask about real companies, or **📈 Results** for the numbers.")


def _try_it() -> None:
    st.markdown("Ask about real public companies — the agent fetches their actual SEC filings.")
    st.session_state.setdefault("question", _EXAMPLES[0])

    cols = st.columns(len(_EXAMPLES))
    for i, example in enumerate(_EXAMPLES):
        cols[i].button(
            f"Example {i + 1}", key=f"ex{i}", help=example, on_click=_set_question, args=(example,)
        )

    if st.button("💡 Suggest questions"):
        with st.spinner("Thinking of good questions…"):
            try:
                result = LLMClient().chat_structured(
                    [{"role": "user", "content": _SUGGEST_PROMPT}], _Suggestions
                )
                st.session_state.suggestions = result.questions
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Couldn't suggest questions: {exc}")
    for i, suggestion in enumerate(st.session_state.get("suggestions", [])):
        st.button(f"💡 {suggestion}", key=f"sug{i}", on_click=_set_question, args=(suggestion,))

    task = st.text_area("Question", key="question", height=80)
    st.caption("Hosted demo covers **Apple, Microsoft, NVIDIA** (bundled from real filings).")

    if st.button("Research it ✨", type="primary"):
        status = st.status(
            "🔎 Researching — reading filings, looking up figures, computing…", expanded=True
        )
        emojis = {"xbrl_value": "📄", "compute": "🧮", "passage": "📝"}

        def _on_step(step: object) -> None:
            mark = "✅" if step.ok else "⚠️"
            status.markdown(f"{mark} {emojis.get(step.tool, '•')} **{step.tool}** · {step.target}")
            status.caption(step.observation[:140])

        try:
            agent = ResearchAgent(LLMClient(), make_dispatch(_registry()), max_steps=6)
            report = agent.run(task, on_step=_on_step)
        except Exception as exc:  # noqa: BLE001 - surface provider/network errors to the user
            status.update(label="⚠️ Couldn't finish", state="error")
            st.error(f"Something went wrong: {exc}")
            st.stop()

        status.update(label="✅ Done", state="complete")
        st.balloons()
        st.success(f"### {report.summary}")
        if report.trend and report.trend.lower() != "n/a":
            st.caption(f"Verdict: **{report.trend}**")
        tel = report.telemetry
        st.caption(
            f"✨ {tel.tool_calls} tool calls · {tel.succeeded} verified · "
            f"{tel.corrections} self-correction(s)"
        )


def _results() -> None:
    st.markdown("#### Determinism — who should do the math?")
    st.caption("Same model, same evidence; only *who computes* differs. FinQA, gpt-4.1-mini.")
    determinism = pd.DataFrame(
        {"Program + executor": [60.3, 51.9], "LLM does the math": [69.1, 44.2]},
        index=["single-step", "multi-step"],
    )
    st.bar_chart(determinism)
    st.caption(
        "On multi-step arithmetic the executor wins by ~8 points — determinism pays off exactly "
        "where mental math breaks down."
    )

    st.divider()
    st.markdown("#### Retrieval — hybrid beats either approach alone")
    retrieval = pd.DataFrame(
        {"recall@5 (%)": [81.9, 82.6, 85.6]}, index=["BM25", "dense", "hybrid"]
    )
    st.bar_chart(retrieval)
    st.caption("FinQA dev, n=150. BM25 anchors exact matches; dense adds recall; RRF fuses both.")

    st.divider()
    st.markdown("#### Trust gates")
    g1, g2 = st.columns(2)
    g1.metric("Answer precision", "64.5%", "+4.5 vs no gates")
    g2.metric("Correct answers wrongly declined", "0%")
    st.caption("The gates raise precision by abstaining only on answers they can't verify.")


def _how_built() -> None:
    st.markdown("#### In plain English")
    st.markdown(
        "Picture a careful analyst who is **not allowed to do mental math** and must **cite every "
        "number.** Ask a question and they (1) find the relevant lines in the filing, (2) write "
        "down the exact figures with their source, (3) punch the calculation into a calculator, "
        "and (4) sanity-check the result before answering — or say *\"I can't verify that\"* "
        "instead of guessing. That's the whole design."
    )

    st.divider()
    st.markdown("#### The architecture")
    st.graphviz_chart(
        r"""
        digraph {
            rankdir=LR; bgcolor="transparent"; node [shape=box style=rounded fontsize=11];
            Q [label="Question"]; R [label="Retrieve evidence\n(BM25 + dense)"];
            P [label="LLM proposes\na PROGRAM"]; X [label="Executor\nruns the math"];
            G [label="Trust gates" shape=diamond]; A [label="Answer + citations"];
            B [label="Abstain"];
            Q -> R -> P -> X -> G; G -> A [label="pass"]; G -> B [label="fail"];
        }
        """
    )
    st.markdown(
        "The **agent** wraps this as a tool: in a *plan → act → observe → revise* loop it picks "
        "`xbrl_value` (exact filed figure) / `compute` (calculator) / `passage` (text) one step "
        "at a time, across one or more companies — self-correcting when a step fails."
    )

    st.divider()
    st.markdown("#### Walk through one question")
    st.markdown(
        "*\"How fast did Apple's revenue grow last year?\"*\n"
        "1. **Look up** Apple's revenue for the two most recent years — pulled straight from its "
        "official filed data, not guessed.\n"
        "2. **Compute** the growth with a real calculator: `(new − old) / old`.\n"
        "3. **Check** every input traces to a real figure and the result is sensible.\n"
        "4. **Answer** with the number and its source — or abstain if anything didn't check out."
    )

    st.divider()
    st.markdown("#### Why the design is sound")
    st.markdown(
        "- **The AI can't invent numbers** — it only chooses *which* figures to use and *what* "
        "to calculate; a deterministic engine does the arithmetic.\n"
        "- **Every figure is grounded** — each traces to the filing or the company's filed data, "
        "so a fabricated number is caught.\n"
        "- **Knows its limits** — it abstains rather than answer on shaky ground.\n"
        "- **It's measured, not asserted** — the calculator is proven correct on 8,281 expert "
        "problems; the gates and retrieval each have hard numbers (see **Results**)."
    )

    st.divider()
    st.markdown("#### Scope & honesty")
    st.markdown(
        "This is a **research prototype of a *verifiable-AI* architecture**, not a general "
        "financial oracle. It's evaluated on the academic **FinQA** benchmark and demonstrated "
        "live on real SEC 10-Ks. The point isn't breadth — it's a **sound, auditable pattern** "
        "for making an LLM's numbers trustworthy, one that transfers to any domain where answers "
        "must be *computed and cited*, not guessed."
    )

    st.divider()
    st.markdown("#### Under the hood")
    st.markdown(
        "- **Executor** — pure-Python interpreter for a small program DSL (*Program-of-Thoughts*); "
        "the LLM never does arithmetic. Reproduces 99.5% of 8,281 FinQA gold programs.\n"
        "- **Ingestion** — SEC EDGAR HTML/iXBRL parsing + XBRL facts (ground truth).\n"
        "- **Retrieval** — row-level chunks of real 10-K tables; BM25 + dense, RRF fusion.\n"
        "- **Gates** — operand-grounding, program-validity, numeric-sanity → answer or abstain.\n"
        "- **Agent** — hand-rolled ReAct loop (no LangChain); self-correcting; multi-company.\n"
        "- **Stack** — Azure OpenAI, FastAPI + Streamlit, ruff + pytest + CI; 71 tests."
    )
    st.caption(f"Source: [{_REPO}]({_REPO})")


st.set_page_config(page_title="Agentic Financial Analyst", page_icon="📊", layout="wide")

with st.sidebar:
    st.header("📊 Agentic Financial Analyst")
    st.markdown(
        "Grounded financial-QA over real SEC filings. The AI plans; a calculator computes; "
        "every number is cited and verified."
    )
    st.caption(f"FinQA + live SEC EDGAR · Azure gpt-4.1-mini · [GitHub]({_REPO})")

st.title("📊 Agentic Financial Analyst")
tab_overview, tab_try, tab_results, tab_built = st.tabs(
    ["🏠 Overview", "🔎 Try it", "📈 Results", "🛠 How it's built"]
)
with tab_overview:
    _overview()
with tab_try:
    _try_it()
with tab_results:
    _results()
with tab_built:
    _how_built()
