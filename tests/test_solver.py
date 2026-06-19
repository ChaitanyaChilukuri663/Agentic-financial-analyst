"""Mocked tests for the extract+plan solver and determinism baseline (no LLM/network)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ledgerlens.evaluation.finqa import FinqaExample
from ledgerlens.evaluation.finqa_qa import run_determinism_baseline
from ledgerlens.executor.dsl import Table
from ledgerlens.executor.numbers import answers_match
from ledgerlens.solver.schema import DirectAnswer, ProgramProposal
from ledgerlens.solver.solver import solve_direct, solve_program


class _StubClient:
    """Returns a fixed response from chat_structured, recording the calls."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[Any] = []

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, response_model))
        return self._response


def _proposal(program: str) -> ProgramProposal:
    return ProgramProposal(reasoning="r", operands=[], program=program)


def test_solve_program_parses_and_executes() -> None:
    client = _StubClient(_proposal("subtract(5829, 5735), divide(#0, 5735)"))
    result = solve_program(client, "q?", "evidence")
    assert result.ok
    assert answers_match(result.answer, 0.016390)


def test_solve_program_reports_parse_failure() -> None:
    client = _StubClient(_proposal("this is not a program"))
    result = solve_program(client, "q?", "evidence")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("parse")


def test_solve_program_with_table() -> None:
    client = _StubClient(_proposal("table_sum(revenue, none)"))
    table = Table(rows=[["revenue", "100", "200"]])
    result = solve_program(client, "q?", "evidence", table=table)
    assert result.ok
    assert result.answer == Decimal("300")


def test_solve_program_reports_exec_failure() -> None:
    client = _StubClient(_proposal("divide(1, 0)"))
    result = solve_program(client, "q?", "evidence")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("exec_error")


def test_solve_direct_returns_answer_string() -> None:
    client = _StubClient(DirectAnswer(reasoning="r", answer="0.0164"))
    assert solve_direct(client, "q?", "evidence") == "0.0164"


class _ModeStub:
    """Returns a program proposal or a direct answer depending on the requested model."""

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> Any:
        if response_model is ProgramProposal:
            return _proposal("subtract(5829, 5735), divide(#0, 5735)")
        return DirectAnswer(reasoning="r", answer="0.0164")


def test_determinism_baseline_tally() -> None:
    example = FinqaExample(
        uid="x",
        program="",
        gold_answer=0.016390,
        table=Table(rows=[]),
        question="percent change?",
        pre_text="net revenue was 5829 and 5735",
    )
    report = run_determinism_baseline(_ModeStub(), [example])
    assert report.total == 1
    assert report.program_correct == 1
    assert report.direct_correct == 1
