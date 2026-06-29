"""Mocked tests for the research agent loop (no LLM / network)."""

from __future__ import annotations

from typing import Any

from ledgerlens.agent.agent import ResearchAgent
from ledgerlens.agent.schema import ReportSynthesis, ResearchPlan
from ledgerlens.agent.tools import VerifiedCalculator
from ledgerlens.solver.schema import ProgramProposal


class _ModeStub:
    """Returns a plan, a program, or a synthesis depending on the requested schema."""

    def __init__(self, subquestions: list[str], program: str) -> None:
        self._subquestions = subquestions
        self._program = program

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> Any:
        if response_model is ResearchPlan:
            return ResearchPlan(subquestions=self._subquestions)
        if response_model is ReportSynthesis:
            return ReportSynthesis(summary="Revenue grew.", trend="increasing")
        return ProgramProposal(reasoning="r", operands=[], program=self._program)


def _retrieve(question: str, k: int) -> str:
    return "net revenue was 5829 in 2023 and 5735 in 2022"


def test_agent_plans_computes_and_synthesizes() -> None:
    stub = _ModeStub(
        subquestions=["change in net revenue 2022 to 2023?"],
        program="subtract(5829, 5735)",
    )
    agent = ResearchAgent(stub, VerifiedCalculator(stub, _retrieve))
    report = agent.run("How did net revenue change?")
    assert report.telemetry.subquestions == 1
    assert report.telemetry.answered == 1
    assert report.findings[0].answer is not None
    assert report.summary == "Revenue grew."
    assert report.trend == "increasing"


def test_agent_abstains_on_ungrounded_finding() -> None:
    stub = _ModeStub(subquestions=["q"], program="subtract(9999, 5735)")  # 9999 not in evidence
    agent = ResearchAgent(stub, VerifiedCalculator(stub, _retrieve), recover=False)
    report = agent.run("task")
    assert report.telemetry.abstained == 1
    assert not report.findings[0].answered
