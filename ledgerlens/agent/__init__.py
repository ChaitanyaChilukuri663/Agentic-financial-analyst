"""Agentic research analyst (P6): orchestrates LedgerLens as a verified-computation tool.

A hand-rolled plan -> act -> observe -> revise -> synthesize loop. Every figure in the
final report comes from a gated, cited tool call — an agent that can't fabricate its numbers.
"""

from ledgerlens.agent.agent import AgentTelemetry, Report, ResearchAgent
from ledgerlens.agent.schema import ReportSynthesis, ResearchPlan
from ledgerlens.agent.tools import Finding, VerifiedCalculator

__all__ = [
    "AgentTelemetry",
    "Finding",
    "Report",
    "ReportSynthesis",
    "ResearchAgent",
    "ResearchPlan",
    "VerifiedCalculator",
]
