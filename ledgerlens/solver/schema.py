"""LLM-facing schemas for the extract+plan step (P2).

The LLM proposes a *program string* (FinQA-style) plus the operands it used, each
cited to the evidence — never the final number. We parse the string into the DSL and
the deterministic executor computes the result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Operand(BaseModel):
    """A number used by the program, tied to its source in the evidence."""

    name: str = Field(description="What the number represents, e.g. 'net revenue 2023'.")
    value: str = Field(description="The value verbatim from the evidence, e.g. '5,829'.")
    citation: str = Field(description="Short quote or table row where the value appears.")


class ProgramProposal(BaseModel):
    """A proposed reasoning program (operations over operands) — not the answer."""

    reasoning: str = Field(description="Brief reasoning for the steps (not executed).")
    operands: list[Operand] = Field(description="Operands used, each cited to the evidence.")
    program: str = Field(
        description="Program string, e.g. 'subtract(5829, 5735), divide(#0, 5735)'."
    )


class DirectAnswer(BaseModel):
    """Baseline mode: the LLM computes the answer directly (no executor)."""

    reasoning: str = Field(description="Brief reasoning.")
    answer: str = Field(
        description="Final answer: a number (e.g. '0.0164', '14.1%') or 'yes'/'no'."
    )
