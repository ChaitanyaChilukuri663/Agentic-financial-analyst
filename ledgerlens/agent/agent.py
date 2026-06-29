"""The research agent: a hand-rolled ReAct loop over verified tools (P7).

No framework. At each step the LLM picks ONE action (calculate / xbrl_value / passage /
finish) given the task and the running progress; tools execute deterministically and their
observations feed the next decision. The loop is **iterative** (each step chosen from the
current state), **self-correcting** (it sees failures and reformulates), **multi-tool**, and
**multi-company**. Every number in the final answer came from a tool — the agent cannot
fabricate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ledgerlens.agent.prompts import build_step_messages
from ledgerlens.agent.schema import AgentAction
from ledgerlens.agent.tools import Dispatch, ToolResult
from ledgerlens.llm import LLMClient


@dataclass
class AgentStep:
    """One taken step: the thought, the tool + target, and the observation."""

    thought: str
    tool: str
    target: str
    observation: str
    ok: bool


@dataclass
class AgentTelemetry:
    """Run-level signals for the agent eval."""

    steps: int = 0
    tool_calls: int = 0
    succeeded: int = 0
    failed: int = 0
    corrections: int = 0  # a success immediately after a failed step (recovery)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.tool_calls if self.tool_calls else 0.0


@dataclass
class Report:
    """The agent's final answer plus the full, inspectable trace."""

    task: str
    summary: str
    trend: str
    steps: list[AgentStep] = field(default_factory=list)
    telemetry: AgentTelemetry = field(default_factory=AgentTelemetry)


def _progress(steps: list[AgentStep]) -> str:
    return "\n".join(f"- {s.tool}({s.target}) -> {s.observation}" for s in steps)


def _action_key(action: AgentAction) -> str:
    fields = (
        action.tool,
        action.company,
        action.query,
        action.fiscal_year,
        action.operation,
        tuple(action.values),
    )
    return "|".join(str(field) for field in fields)


def _to_step(action: AgentAction, result: ToolResult) -> AgentStep:
    if result.ok:
        observation = result.output + (f" [{result.citation}]" if result.citation else "")
    else:
        observation = f"FAILED: {result.note}"
    if action.tool == "compute":
        target = f"{action.operation}({', '.join(str(v) for v in action.values)})"
    else:
        target = f"{action.company}:{action.query}".strip(":")
    return AgentStep(action.thought, action.tool, target, observation, result.ok)


class ResearchAgent:
    """Iterative, self-correcting, multi-tool, multi-company ReAct agent over LedgerLens."""

    def __init__(self, client: LLMClient, dispatch: Dispatch, *, max_steps: int = 8) -> None:
        self.client = client
        self.dispatch = dispatch
        self.max_steps = max_steps

    def run(self, task: str) -> Report:
        steps: list[AgentStep] = []
        telemetry = AgentTelemetry()
        prev_failed = False
        seen: set[str] = set()
        for _ in range(self.max_steps):
            action = self.client.chat_structured(
                build_step_messages(task, _progress(steps)), AgentAction
            )
            telemetry.steps += 1
            if action.tool == "finish":
                return Report(task, action.summary, action.trend, steps, telemetry)
            key = _action_key(action)
            if key in seen:
                note = "already tried this action; change approach"
                steps.append(_to_step(action, ToolResult(action.tool, False, "", note=note)))
                prev_failed = True
                continue
            seen.add(key)
            result = self.dispatch(action)
            telemetry.tool_calls += 1
            telemetry.succeeded += int(result.ok)
            telemetry.failed += int(not result.ok)
            if result.ok and prev_failed:
                telemetry.corrections += 1
            prev_failed = not result.ok
            steps.append(_to_step(action, result))
        # Step budget reached — force a final answer from what we have.
        final = self.client.chat_structured(
            build_step_messages(task, _progress(steps) + "\n(Step budget reached — finish now.)"),
            AgentAction,
        )
        summary = final.summary or "Reached the step limit before a final answer."
        return Report(task, summary, final.trend, steps, telemetry)
