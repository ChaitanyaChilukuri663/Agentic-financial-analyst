"""LedgerLens end-to-end engine: question + evidence -> verified, cited answer.

Ties the pieces together — the LLM proposes a program (solver), the deterministic
executor computes it, and the gates decide answer-or-abstain. The returned
:class:`Answer` carries everything the UI / API render: the computed answer, the shown
program steps, the cited operands, and any abstention reasons.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from ledgerlens.executor.dsl import Arg, NoneArg, NumberArg, RefArg, RowArg, Table
from ledgerlens.executor.numbers import str_to_num
from ledgerlens.llm import LLMClient
from ledgerlens.solver.schema import Operand
from ledgerlens.solver.solver import SolveResult, solve_program
from ledgerlens.validation import validate


@dataclass
class StepTrace:
    """One executed program step, rendered for display."""

    op: str
    args: list[str]
    result: str


@dataclass
class Answer:
    """A fully-traced answer: the number, the shown work, the citations, the gate verdict."""

    question: str
    answered: bool
    answer: str | None
    abstain_reasons: list[str]
    program: str | None
    operands: list[Operand]
    steps: list[StepTrace]


def answer_question(
    client: LLMClient,
    question: str,
    context: str,
    *,
    table: Table | None = None,
    parse_cell: Callable[[str], Decimal | None] = str_to_num,
) -> Answer:
    """Propose a program, execute it, gate it, and return a cited, traced answer."""
    result = solve_program(client, question, context, table=table, parse_cell=parse_cell)
    verdict = validate(result, context)
    steps = _trace(result) if result.ok and result.program else []
    return Answer(
        question=question,
        answered=verdict.accept,
        answer=_format(result.answer) if verdict.accept else None,
        abstain_reasons=verdict.abstain_reasons,
        program=result.proposal.program if result.proposal else None,
        operands=result.proposal.operands if result.proposal else [],
        steps=steps,
    )


def _trace(result: SolveResult) -> list[StepTrace]:
    traces: list[StepTrace] = []
    program = result.program
    if program is None:
        return traces
    for step, value in zip(program.steps, result.steps, strict=False):
        traces.append(
            StepTrace(op=step.op.value, args=[_arg(a) for a in step.args], result=_format(value))
        )
    return traces


def _arg(arg: Arg) -> str:
    if isinstance(arg, NumberArg):
        return _format(arg.value)
    if isinstance(arg, RefArg):
        return f"#{arg.step}"
    if isinstance(arg, RowArg):
        return arg.name
    if isinstance(arg, NoneArg):
        return "none"
    return "?"


def _format(value: Decimal | bool | None) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Decimal):
        return f"{value.normalize():f}"
    return "" if value is None else str(value)
