"""Mocked tests for the ReAct agent loop + the verified compute tool (no LLM)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from ledgerlens.agent.agent import ResearchAgent
from ledgerlens.agent.schema import AgentAction
from ledgerlens.agent.tools import ToolResult, tool_compute, tool_xbrl_value
from ledgerlens.agent.workspace import FilingWorkspace, load_workspace_bundle
from ledgerlens.ingest.xbrl import XbrlFact
from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk


class _ScriptedClient:
    """Returns a pre-scripted sequence of AgentActions (one per step)."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = actions
        self._i = 0

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> AgentAction:
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return action


def _action(tool: str, **kwargs: Any) -> AgentAction:
    return AgentAction(thought="t", tool=tool, **kwargs)


def test_agent_looks_up_then_finishes() -> None:
    client = _ScriptedClient(
        [
            _action("xbrl_value", company="AAPL", query="revenue"),
            _action("finish", summary="Apple's latest revenue is $416B.", trend="n/a"),
        ]
    )
    outcomes = iter([ToolResult("xbrl_value", True, "416161000000", "Revenues FY2025")])

    def dispatch(action: AgentAction) -> ToolResult:
        return next(outcomes)

    report = ResearchAgent(client, dispatch, max_steps=5).run("Apple's latest revenue?")
    assert report.summary == "Apple's latest revenue is $416B."
    assert report.telemetry.tool_calls == 1
    assert report.telemetry.succeeded == 1
    assert len(report.steps) == 1


def test_agent_self_corrects_after_a_failure() -> None:
    client = _ScriptedClient(
        [
            _action("xbrl_value", company="AAPL", query="sales"),
            _action("xbrl_value", company="AAPL", query="revenue"),
            _action("finish", summary="Revenue was 416,161M.", trend="n/a"),
        ]
    )
    outcomes = iter(
        [
            ToolResult("xbrl_value", False, "", note="no XBRL fact for 'sales'"),
            ToolResult("xbrl_value", True, "416161000000", "Revenues FY2025"),
        ]
    )

    def dispatch(action: AgentAction) -> ToolResult:
        return next(outcomes)

    report = ResearchAgent(client, dispatch, max_steps=5).run("Apple revenue?")
    assert report.telemetry.failed == 1
    assert report.telemetry.succeeded == 1
    assert report.telemetry.corrections == 1  # recovered right after the failure


def test_agent_abstains_after_repeated_failures() -> None:
    client = _ScriptedClient(
        [
            _action("xbrl_value", company="AAPL", query="inventory"),
            _action("xbrl_value", company="AAPL", query="stock on hand"),
            _action("xbrl_value", company="AAPL", query="widgets"),
            _action("xbrl_value", company="AAPL", query="gadgets"),
        ]
    )

    def dispatch(action: AgentAction) -> ToolResult:
        return ToolResult(action.tool, False, "", note="no XBRL fact")

    report = ResearchAgent(client, dispatch, max_steps=8).run("Apple's inventory?")
    assert report.trend == "abstained"
    assert "couldn't find" in report.summary.lower()
    assert report.telemetry.failed >= 3


def test_compute_uses_executor_and_grounds_to_prior_figures() -> None:
    ledger = [Decimal("143015000000"), Decimal("168088000000")]
    ok = tool_compute("percent_change", [143015000000, 168088000000], ledger)
    assert ok.ok
    assert abs(Decimal(ok.output) - Decimal("0.17532")) < Decimal("0.001")

    fabricated = tool_compute("percent_change", [143015000000, 999000000000], ledger)
    assert not fabricated.ok  # 999B was never returned by a tool


def _fact(concept: str, year: int, value: str) -> XbrlFact:
    return XbrlFact(
        taxonomy="us-gaap",
        concept=concept,
        unit="USD",
        value=Decimal(value),
        period_end=f"{year}-12-31",
        fiscal_year=year,
        fiscal_period="FY",
        form="10-K",
    )


def test_xbrl_value_pools_alias_concepts_for_latest_year() -> None:
    # A company can report revenue under different us-gaap concepts across years (NVDA does):
    # the ASC-606 concept stops at 2022 while `Revenues` carries the newer figures. The lookup
    # must return the truly latest value, not the latest of whichever concept it checks first.
    facts = [
        _fact("RevenueFromContractWithCustomerExcludingAssessedTax", 2021, "16675"),
        _fact("RevenueFromContractWithCustomerExcludingAssessedTax", 2022, "26914"),
        _fact("Revenues", 2024, "60922"),
        _fact("Revenues", 2026, "215938"),
    ]
    ws = FilingWorkspace(
        ticker="NVDA",
        cik="1045810",
        title="NVIDIA CORP",
        filing_url="https://example.com",
        bm25=Bm25Index([Chunk(chunk_id="c0", text="placeholder", kind="text")]),
        facts=facts,
    )
    latest = tool_xbrl_value(ws, "revenue", None)
    assert latest.ok
    assert latest.output == "215938"  # Revenues FY2026, not the stale ASC-606 FY2022
    specific = tool_xbrl_value(ws, "revenue", 2022)
    assert specific.ok
    assert specific.output == "26914"  # per-year lookups still resolve the right concept


def test_committed_demo_bundle_loads() -> None:
    bundle = (
        Path(__file__).resolve().parents[1] / "ledgerlens" / "agent" / "demo_data" / "AAPL.json"
    )
    if not bundle.exists():
        return  # bundles are optional; skip if not generated
    workspace = load_workspace_bundle(bundle)
    assert workspace.ticker == "AAPL"
    assert len(workspace.facts) > 0
