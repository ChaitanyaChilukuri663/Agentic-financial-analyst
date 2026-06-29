"""Mocked tests for the ReAct agent loop + the verified compute tool (no LLM)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ledgerlens.agent.agent import ResearchAgent
from ledgerlens.agent.schema import AgentAction
from ledgerlens.agent.tools import ToolResult, tool_compute


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


def test_compute_uses_executor_and_grounds_to_prior_figures() -> None:
    ledger = [Decimal("143015000000"), Decimal("168088000000")]
    ok = tool_compute("percent_change", [143015000000, 168088000000], ledger)
    assert ok.ok
    assert abs(Decimal(ok.output) - Decimal("0.17532")) < Decimal("0.001")

    fabricated = tool_compute("percent_change", [143015000000, 999000000000], ledger)
    assert not fabricated.ok  # 999B was never returned by a tool
