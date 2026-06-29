"""LLM-facing schemas for the research agent (P6)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """A decomposition of the task into individually-answerable sub-questions."""

    subquestions: list[str] = Field(
        description="Specific sub-questions, each asking for ONE figure or ONE calculation."
    )


class ReportSynthesis(BaseModel):
    """The analyst's final answer, grounded only in the verified findings."""

    summary: str = Field(description="Concise answer to the task, citing the computed figures.")
    trend: str = Field(description="The observed trend/direction (e.g. 'increasing'), or 'n/a'.")
