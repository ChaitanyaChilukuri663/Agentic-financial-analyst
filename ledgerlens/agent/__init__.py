"""Agentic research analyst (P7): an iterative, self-correcting, multi-tool ReAct loop.

The agent orchestrates LedgerLens across one or more companies' 10-Ks. At each step it picks a
tool — verified calculation, exact XBRL lookup, or passage lookup — observes the result, and
adapts. Every figure in the final report came from a gated, cited tool call: an agent that
can't fabricate its numbers.
"""

from ledgerlens.agent.agent import AgentStep, AgentTelemetry, Report, ResearchAgent
from ledgerlens.agent.schema import AgentAction
from ledgerlens.agent.tools import ToolResult, make_dispatch
from ledgerlens.agent.workspace import FilingWorkspace, WorkspaceRegistry

__all__ = [
    "AgentAction",
    "AgentStep",
    "AgentTelemetry",
    "FilingWorkspace",
    "Report",
    "ResearchAgent",
    "ToolResult",
    "WorkspaceRegistry",
    "make_dispatch",
]
