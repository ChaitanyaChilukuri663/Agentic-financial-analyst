"""The program DSL: a symbolic, JSON-serializable reasoning program.

A :class:`Program` is an ordered list of :class:`Step`. Each step applies an
:class:`Op` to a list of args. An arg is a literal number, a reference to a prior
step's result (``#n``), a table row name, or the FinQA ``none`` placeholder.

This is exactly the form the LLM will emit in P2 (Pydantic-validated) and the form
the deterministic executor consumes — so the contract lives in one place.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Op(StrEnum):
    """The supported operations (a superset of FinQA's op set)."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    EXP = "exp"
    GREATER = "greater"
    TABLE_SUM = "table_sum"
    TABLE_AVERAGE = "table_average"
    TABLE_MAX = "table_max"
    TABLE_MIN = "table_min"


MATH_OPS = frozenset({Op.ADD, Op.SUBTRACT, Op.MULTIPLY, Op.DIVIDE, Op.EXP, Op.GREATER})
TABLE_OPS = frozenset({Op.TABLE_SUM, Op.TABLE_AVERAGE, Op.TABLE_MAX, Op.TABLE_MIN})


class NumberArg(BaseModel):
    """A literal numeric operand."""

    kind: Literal["number"] = "number"
    value: Decimal = Field(description="The literal numeric value.")


class RefArg(BaseModel):
    """A reference to the result of a prior step (FinQA ``#n``)."""

    kind: Literal["ref"] = "ref"
    step: int = Field(description="Zero-based index of the prior step whose result is reused.")


class RowArg(BaseModel):
    """A table row name, used as the first operand of a ``table_*`` op."""

    kind: Literal["row"] = "row"
    name: str = Field(description="Row header to look up in the table.")


class NoneArg(BaseModel):
    """The FinQA ``none`` placeholder (second operand of table ops)."""

    kind: Literal["none"] = "none"


Arg = Annotated[NumberArg | RefArg | RowArg | NoneArg, Field(discriminator="kind")]


class Step(BaseModel):
    """One operation applied to its operands."""

    op: Op = Field(description="The operation to apply.")
    args: list[Arg] = Field(description="Operands, in order.")


class Program(BaseModel):
    """An ordered reasoning program; the last step's result is the answer."""

    steps: list[Step] = Field(description="Reasoning steps, executed in order.")


class Table(BaseModel):
    """A structured table; each row is a list of cell strings, ``rows[i][0]`` the label."""

    rows: list[list[str]] = Field(default_factory=list, description="Raw table rows.")

    def find_row(self, name: str) -> list[str] | None:
        """Return the first row whose label matches ``name`` (case/space-insensitive)."""
        target = _normalize(name)
        for row in self.rows:
            if row and _normalize(row[0]) == target:
                return row
        return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
