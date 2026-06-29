"""The research agent: a hand-rolled plan -> act -> observe -> revise -> synthesize loop.

No framework (no LangChain). The agent plans sub-questions, answers each through the
verified calculator (retrieve + gated compute), recovers once from an abstention by widening
retrieval, then synthesizes a report whose every figure came from a verified tool call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ledgerlens.agent.prompts import build_plan_messages, build_synth_messages
from ledgerlens.agent.schema import ReportSynthesis, ResearchPlan
from ledgerlens.agent.tools import Finding, VerifiedCalculator
from ledgerlens.llm import LLMClient


@dataclass
class AgentTelemetry:
    """Run-level signals for the agent eval."""

    subquestions: int = 0
    tool_calls: int = 0
    answered: int = 0
    abstained: int = 0
    recoveries: int = 0

    @property
    def grounded_rate(self) -> float:
        return self.answered / self.subquestions if self.subquestions else 0.0


@dataclass
class Report:
    """The agent's final, citation-grounded report."""

    task: str
    summary: str
    trend: str
    findings: list[Finding] = field(default_factory=list)
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


class ResearchAgent:
    """Orchestrates LedgerLens across a multi-step task; numbers are always tool-verified."""

    def __init__(
        self, client: LLMClient, calculator: VerifiedCalculator, *, recover: bool = True
    ) -> None:
        self.client = client
        self.calc = calculator
        self.recover = recover

    def run(self, task: str) -> Report:
        plan = self.client.chat_structured(build_plan_messages(task), ResearchPlan)
        telemetry = AgentTelemetry(subquestions=len(plan.subquestions))
        findings: list[Finding] = []
        for question in plan.subquestions:
            finding = self.calc.compute(question)
            telemetry.tool_calls += 1
            if not finding.answered and self.recover:
                wider = self.calc.compute(question, k=16)  # recover: widen retrieval, retry once
                telemetry.tool_calls += 1
                if wider.answered:
                    finding = wider
                    telemetry.recoveries += 1
            findings.append(finding)
            telemetry.answered += int(finding.answered)
            telemetry.abstained += int(not finding.answered)
        synthesis = self._synthesize(task, findings)
        return Report(
            task=task,
            summary=synthesis.summary,
            trend=synthesis.trend,
            findings=findings,
            telemetry=telemetry,
        )

    def _synthesize(self, task: str, findings: list[Finding]) -> ReportSynthesis:
        lines: list[str] = []
        for finding in findings:
            if finding.answered:
                citation = finding.citations[0] if finding.citations else ""
                lines.append(f"- {finding.question} -> {finding.answer}  [{citation}]")
            else:
                lines.append(f"- {finding.question} -> (insufficient data)")
        return self.client.chat_structured(
            build_synth_messages(task, "\n".join(lines)), ReportSynthesis
        )
