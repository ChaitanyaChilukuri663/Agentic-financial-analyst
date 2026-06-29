"""LLM-facing schema for the ReAct agent (P7): one action per step."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """One ReAct step: a thought plus the next tool to call (or 'finish')."""

    thought: str = Field(description="Brief reasoning about what to do next, given what's known.")
    tool: Literal["xbrl_value", "compute", "passage", "finish"] = Field(
        description="Which tool to call, or 'finish' to end with the answer."
    )
    company: str = Field(default="", description="Ticker for xbrl_value / passage, e.g. AAPL.")
    query: str = Field(
        default="",
        description="xbrl_value: a concept keyword (e.g. 'revenue'). passage: the question.",
    )
    fiscal_year: int | None = Field(
        default=None, description="xbrl_value: the fiscal year, if any."
    )
    operation: str = Field(
        default="", description="compute: difference | percent_change | ratio | sum | average."
    )
    values: list[float] = Field(
        default_factory=list,
        description="compute: figures to operate on (must be ones the tools already returned).",
    )
    summary: str = Field(
        default="", description="finish: the final answer, citing the figures found."
    )
    trend: str = Field(default="", description="finish: the observed trend/verdict, or 'n/a'.")
