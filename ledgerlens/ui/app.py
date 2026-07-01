"""Streamlit demo for the Agentic Financial Analyst.

Five tabs, framed for non-technical visitors (recruiters, HR):
  Overview       — plain-English: what it is, why it beats a chatbot, why it was built.
  Try it         — ask the agent about real companies; see the proof behind every number.
  vs ChatGPT     — the same question through a plain chatbot vs this system, side by side.
  Results        — the evaluation numbers as charts.
  How it's built — the architecture diagram + the stack.

Local:           streamlit run ledgerlens/ui/app.py   (reads .env)
Streamlit Cloud: set LEDGERLENS_* secrets in the dashboard (bridged to env below).
"""

from __future__ import annotations

import os
import re
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

# Short labels keep the example buttons readable for non-technical visitors.
_EXAMPLES = [
    ("📈 Revenue growth",
     "How fast did Apple's (AAPL) revenue grow in its most recent fiscal year?"),
    ("🆚 Compare two",
     "Compare NVIDIA (NVDA) and Microsoft (MSFT) on revenue growth — which grew faster?"),
    ("🧮 Profit margin",
     "What was Apple's (AAPL) net profit margin in its most recent fiscal year?"),
    ("💰 Net income", "What was Amazon's (AMZN) net income in its most recent fiscal year?"),
    ("🛑 Watch it abstain", "What was Apple's (AAPL) inventory last year?"),
]
# vs-ChatGPT defaults focus on where a memory-only model visibly fails: recency + hard math.
_VS_EXAMPLES = [
    ("NVIDIA growth", "How fast did NVIDIA's (NVDA) revenue grow in its most recent fiscal year?"),
    ("Amazon revenue", "What was Amazon's (AMZN) revenue in its most recent fiscal year?"),
    ("Apple margin", "What was Apple's (AAPL) net profit margin in its most recent fiscal year?"),
]
_SUGGEST_PROMPT = (
    "Suggest 3 short, specific questions an analyst could ask about a large public company's "
    "most recent annual financials (e.g. revenue growth, margin, net income), using real "
    "tickers like AAPL, MSFT, NVDA, GOOGL, AMZN, or META."
)
_RAW_SYSTEM = (
    "You are a helpful assistant. Answer the finance question concisely, including the specific "
    "number. You have no tools and no documents — answer from your own knowledge."
)

# Companies with committed demo bundles (hosted app answers these with no live SEC calls).
_DEMO_DIR = Path(__file__).resolve().parents[1] / "agent" / "demo_data"
_DEMO_TICKERS = sorted(p.stem for p in _DEMO_DIR.glob("*.json")) if _DEMO_DIR.exists() else []
_REVENUE_CONCEPTS = {
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
}


class _Suggestions(BaseModel):
    """A few suggested questions for the demo."""

    questions: list[str] = Field(
        description="3 short, specific questions about a company's financials."
    )


class _RawAnswer(BaseModel):
    """A plain, memory-only answer from the raw model (the ChatGPT-style baseline)."""

    answer: str = Field(description="A concise answer including the specific number.")


@st.cache_resource
def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(EdgarClient(_UA))


def _set_question(value: str) -> None:
    st.session_state.question = value


def _set_vs_question(value: str) -> None:
    st.session_state.vs_question = value


def _revenue_history(workspace: object) -> dict[int, float]:
    """Revenue (in $B) by fiscal year from a workspace's facts — max-abs per year."""
    by_year: dict[int, float] = {}
    for fact in getattr(workspace, "facts", []):
        if fact.concept in _REVENUE_CONCEPTS and fact.fiscal_year:
            value = abs(float(fact.value)) / 1e9
            if value > by_year.get(fact.fiscal_year, 0.0):
                by_year[fact.fiscal_year] = value
    return dict(sorted(by_year.items()))


def _fmt_observation(observation: str) -> tuple[str, str]:
    """Split a tool observation ('416161000000 [Revenues FY2025 (USD)]') into (value, source)."""
    head, _, tail = observation.partition(" ")
    source = tail.strip().strip("[]")
    try:
        value = float(head.replace(",", ""))
    except ValueError:
        return head, source
    if abs(value) >= 1e9:
        shown = f"${value / 1e9:.2f}B"
    elif abs(value) < 1:
        shown = f"{value * 100:.2f}%"
    else:
        shown = head
    return shown, source


def _raw_ai_answer(task: str) -> str:
    """The ChatGPT-style baseline: one plain model call, from memory, no tools or filing."""
    messages = [{"role": "system", "content": _RAW_SYSTEM}, {"role": "user", "content": task}]
    return LLMClient().chat_structured(messages, _RawAnswer).answer


def _revenue_chart(task: str) -> None:
    """Show a revenue-by-year line chart for any demo companies named in the question."""
    mentioned = [t for t in _DEMO_TICKERS if re.search(rf"\b{t}\b", task, re.IGNORECASE)]
    series: dict[str, dict[int, float]] = {}
    for ticker in mentioned:
        workspace = _registry().get(ticker)
        history = _revenue_history(workspace) if workspace else {}
        if history:
            series[ticker] = history
    if not series:
        return
    st.markdown("##### Revenue by fiscal year ($B)")
    st.line_chart(pd.DataFrame(series).sort_index())
    st.caption("Context from the same filed figures the agent used — not a separate source.")


def _trust_panel(task: str, report: object) -> None:
    """Make trust tangible: link the real SEC filings and list every verified figure used."""
    with st.expander("🔎 Show the proof — sources & verified figures", expanded=True):
        mentioned = [t for t in _DEMO_TICKERS if re.search(rf"\b{t}\b", task, re.IGNORECASE)]
        links = []
        for ticker in mentioned:
            workspace = _registry().get(ticker)
            url = getattr(workspace, "filing_url", "") if workspace else ""
            if url:
                links.append(f"[{ticker} 10-K on SEC EDGAR]({url})")
        if links:
            st.markdown("**Source filing(s):** " + " · ".join(links))
        rows = []
        for step in getattr(report, "steps", []):
            if step.ok and step.tool in ("xbrl_value", "compute"):
                shown, source = _fmt_observation(step.observation)
                suffix = f"  ·  _{source}_" if source else ""
                rows.append(f"- **{step.target}** → `{shown}`{suffix}")
        if rows:
            st.markdown("**Every number used (each from the filing, computed by code):**")
            st.markdown("\n".join(rows))
        st.caption(
            "This is the difference from a chatbot: you can click through to the exact filing and "
            "see that no number was invented."
        )


def _run_and_render(task: str) -> None:
    """Run the agent with live step streaming, then render the answer, chart, and proof."""
    status = st.status(
        "🔎 Researching — reading filings, looking up figures, computing…", expanded=True
    )
    emojis = {"xbrl_value": "📄", "compute": "🧮", "passage": "📝"}

    def _on_step(step: object) -> None:
        mark = "✅" if step.ok else "⚠️"
        status.markdown(f"{mark} {emojis.get(step.tool, '•')} **{step.tool}** · {step.target}")
        status.caption(step.observation[:140])

    try:
        agent = ResearchAgent(LLMClient(), make_dispatch(_registry()), max_steps=10)
        report = agent.run(task, on_step=_on_step)
    except Exception as exc:  # noqa: BLE001 - surface provider/network errors to the user
        status.update(label="⚠️ Couldn't finish", state="error")
        st.error(f"Something went wrong: {exc}")
        return

    if report.trend == "abstained":
        status.update(label="🛑 Abstained", state="complete")
        st.info(f"**{report.summary}**\n\nThis is by design — it won't invent a figure.")
    else:
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
    if report.trend != "abstained":
        _trust_panel(task, report)
    _revenue_chart(task)


def _overview() -> None:
    st.subheader("AI that gives you financial numbers you can actually trust.")
    st.markdown(
        "Ask about a company's financials and get a straight answer where **every number is pulled "
        "from the company's official SEC filing, calculated by code (not guessed by the AI), and "
        "shown with its source.** If it can't verify something, it says so instead of making it up."
    )

    st.markdown("#### But can't ChatGPT already do this?")
    st.markdown(
        "Not reliably. A general chatbot answers from *memory* — it will confidently hand you a "
        "number that's **out of date or simply invented**, with no way to check it. For a "
        "company's financials, a wrong number isn't a small slip — it can mean a bad investment "
        "or a compliance problem. The difference:"
    )
    st.markdown(
        "| | 🤖 Plain chatbot (ChatGPT / Gemini) | ✅ This system |\n"
        "|---|---|---|\n"
        "| Where the number comes from | its memory — may be stale or invented | the company's "
        "**actual filing**, read live |\n"
        "| Shows its source? | No | **Yes — links the filing + exact line** |\n"
        "| Does the math | itself (often wrong on multi-step) | a **calculator** |\n"
        "| When it doesn't know | makes something up | **says so and stops** |"
    )
    st.caption("👉 See this head-to-head on real questions in the **🆚 vs ChatGPT** tab.")

    st.divider()
    st.markdown("#### A simple way to picture it")
    st.markdown(
        "> A regular chatbot is like a **confident intern** who answers instantly but sometimes "
        "makes the numbers up.\n>\n> This is like a **careful analyst** who looks up each figure "
        "in the annual report, does the math on a calculator, and hands you the answer **with "
        "the page it came from** — or admits when they don't have it."
    )

    st.divider()
    st.markdown("#### Why I built it")
    st.markdown(
        "AI assistants are everywhere, but in high-stakes fields like finance, *sounding right* "
        "isn't good enough — you have to be **provably right**. I built this to work through how "
        "you actually make an AI's numbers trustworthy: separate the reasoning from the "
        "arithmetic, "
        "ground every figure in the real document, verify it, and stay honest about the limits. "
        "The result is a pattern that transfers anywhere answers must be **computed and cited, not "
        "guessed** — audit, compliance, healthcare, law."
    )

    st.divider()
    st.markdown("#### Proven, not just claimed")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calculator accuracy", "99.5%", "8,281 expert problems")
    m2.metric("Answers traced to a source", "100%", "on the agent test set")
    m3.metric("Made-up numbers", "0", "by design")
    m4.metric("Companies in live demo", str(len(_DEMO_TICKERS) or 6), "real-filing data")
    st.info(
        "Ask a question in **🔎 Try it**, see the head-to-head in **🆚 vs ChatGPT**, the evidence "
        "in **📈 Results**, or the engineering in **🛠 How it's built**."
    )


def _try_it() -> None:
    st.markdown(
        "Ask about a real company. Watch it look up each figure in the filing, compute with a "
        "calculator, and show you the source — or decline if it can't verify."
    )
    st.session_state.setdefault("question", _EXAMPLES[0][1])

    cols = st.columns(len(_EXAMPLES))
    for i, (label, example) in enumerate(_EXAMPLES):
        cols[i].button(label, key=f"ex{i}", help=example, on_click=_set_question, args=(example,))

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
    covered = ", ".join(_DEMO_TICKERS) if _DEMO_TICKERS else "AAPL, MSFT, NVDA"
    st.caption(
        f"Hosted demo covers **{covered}** (bundled from real filings). Ask for a figure that "
        "isn't in the data and it will decline rather than guess."
    )

    if st.button("Research it ✨", type="primary"):
        _run_and_render(task)


def _vs_chatgpt() -> None:
    st.markdown(
        "The **same question**, two ways: a plain chatbot answering from memory, vs this system "
        "reading the real filing. Watch what changes — especially the source and the latest-year "
        "numbers."
    )
    st.session_state.setdefault("vs_question", _VS_EXAMPLES[0][1])
    cols = st.columns(len(_VS_EXAMPLES))
    for i, (label, example) in enumerate(_VS_EXAMPLES):
        cols[i].button(
            label, key=f"vs{i}", help=example, on_click=_set_vs_question, args=(example,)
        )
    task = st.text_area("Question", key="vs_question", height=70)

    if st.button("Compare ⚔️", type="primary"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🤖 Plain chatbot")
            st.caption("A language model answering from memory — like ChatGPT. No filing/tools.")
            with st.spinner("Thinking…"):
                try:
                    st.write(_raw_ai_answer(task))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"failed: {exc}")
            st.warning(
                "⚠️ **No source.** Answered from training memory — may be **outdated or made up**, "
                "and there's no way to check it."
            )
        with right:
            st.markdown("#### ✅ This system")
            st.caption("Reads the actual 10-K, computes with a calculator, cites every number.")
            report = None
            with st.spinner("Reading the filing…"):
                try:
                    report = ResearchAgent(
                        LLMClient(), make_dispatch(_registry()), max_steps=10
                    ).run(task)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"failed: {exc}")
            if report is not None:
                if report.trend == "abstained":
                    st.info(f"**{report.summary}**")
                else:
                    st.success(report.summary)
                st.caption(f"✨ {report.telemetry.succeeded} figures verified from the filing")
        if report is not None and report.trend != "abstained":
            _trust_panel(task, report)
        st.info(
            "**The takeaway:** the chatbot's number comes from memory with no proof — it can be "
            "stale or invented. This system's number is pulled from the company's official filing, "
            "computed by code, and shown with its source. Same speed; only one is *verifiable*."
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
    st.markdown("#### The agent, benchmarked")
    a1, a2, a3 = st.columns(3)
    a1.metric("Answer accuracy", "19/19", "labelled questions")
    a2.metric("Faithfulness", "100%", "every number sourced")
    a3.metric("Abstention", "4/4", "declines when it should")
    st.caption(
        "23 labelled questions over the demo companies. *Faithfulness* = every number in every "
        "answer traced back to a verified figure — the 'can't fabricate' claim, measured."
    )

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
        "- **Stack** — Azure OpenAI, FastAPI + Streamlit, ruff + pytest + CI; 75 tests."
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
tab_overview, tab_try, tab_vs, tab_results, tab_built = st.tabs(
    ["🏠 Overview", "🔎 Try it", "🆚 vs ChatGPT", "📈 Results", "🛠 How it's built"]
)
with tab_overview:
    _overview()
with tab_try:
    _try_it()
with tab_vs:
    _vs_chatgpt()
with tab_results:
    _results()
with tab_built:
    _how_built()
