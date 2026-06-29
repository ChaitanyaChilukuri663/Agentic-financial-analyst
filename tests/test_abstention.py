"""Mocked tests for the abstention eval (gates + tallying, no LLM)."""

from __future__ import annotations

from typing import Any

from ledgerlens.evaluation.abstention import run_abstention_eval
from ledgerlens.evaluation.finqa import FinqaExample
from ledgerlens.executor.dsl import Table
from ledgerlens.solver.schema import ProgramProposal


class _Stub:
    def __init__(self, program: str) -> None:
        self._program = program

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> ProgramProposal:
        return ProgramProposal(reasoning="r", operands=[], program=self._program)


def _example(program: str, gold: float, pre: str) -> FinqaExample:
    return FinqaExample(
        uid="x",
        program=program,
        gold_answer=gold,
        table=Table(rows=[]),
        question="q?",
        pre_text=pre,
    )


def test_abstention_accepts_grounded_correct_answer() -> None:
    stub = _Stub("subtract(5829, 5735)")
    example = _example("subtract(5829, 5735)", 94.0, "net revenue 5829 and 5735")
    report = run_abstention_eval(stub, [example])
    assert report.total == 1
    assert report.answered == 1
    assert report.answered_correct == 1
    assert report.coverage == 1.0


def test_abstention_declines_fabricated_operand() -> None:
    stub = _Stub("subtract(9999, 5735)")  # 9999 is not in the evidence
    example = _example("subtract(5829, 5735)", 94.0, "net revenue 5829 and 5735")
    report = run_abstention_eval(stub, [example])
    assert report.answered == 0
    assert report.abstained == 1
    assert sum(report.taxonomy.values()) == 1  # wrong answer classified
