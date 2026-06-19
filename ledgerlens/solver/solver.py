"""The extract+plan solver: the LLM proposes a program, the executor computes it.

This is the heart of LedgerLens online flow. The LLM never returns the final number —
it returns a symbolic program, which the deterministic executor (P1) runs. That is what
makes the computation non-hallucinatory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from ledgerlens.executor.dsl import Program, Table
from ledgerlens.executor.finqa import FinqaProgramError, parse_finqa_program
from ledgerlens.executor.numbers import str_to_num
from ledgerlens.executor.program import execute
from ledgerlens.llm import LLMClient
from ledgerlens.solver.prompts import build_direct_messages, build_program_messages
from ledgerlens.solver.schema import DirectAnswer, ProgramProposal


@dataclass
class SolveResult:
    """Outcome of the extract+plan solve: a computed answer, or a structured failure."""

    ok: bool
    answer: Decimal | bool | None = None
    program: Program | None = None
    proposal: ProgramProposal | None = None
    error: str | None = None


def solve_program(
    client: LLMClient,
    question: str,
    context: str,
    *,
    table: Table | None = None,
    parse_cell: Callable[[str], Decimal | None] = str_to_num,
) -> SolveResult:
    """Ask the LLM for a program, parse it, and execute it deterministically."""
    proposal = client.chat_structured(build_program_messages(question, context), ProgramProposal)
    try:
        program = parse_finqa_program(proposal.program)
    except FinqaProgramError as exc:
        return SolveResult(ok=False, proposal=proposal, error=f"parse_error: {exc}")
    result = execute(program, table, parse_cell=parse_cell)
    if not result.ok:
        code = result.error.code if result.error else "unknown"
        return SolveResult(ok=False, program=program, proposal=proposal, error=f"exec_error:{code}")
    return SolveResult(ok=True, answer=result.value, program=program, proposal=proposal)


def solve_direct(client: LLMClient, question: str, context: str) -> str:
    """Baseline: ask the LLM to compute the answer directly (no executor)."""
    answer = client.chat_structured(build_direct_messages(question, context), DirectAnswer)
    return answer.answer
