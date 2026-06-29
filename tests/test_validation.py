"""Tests for the validation / abstention gates."""

from __future__ import annotations

from decimal import Decimal

from ledgerlens.executor.dsl import NumberArg, Op, Program, Step
from ledgerlens.ingest.xbrl import XbrlFact
from ledgerlens.solver.solver import SolveResult
from ledgerlens.validation import evidence_numbers, is_grounded, validate

_EVIDENCE = "net revenue was $5,829 in 2023 and 5,735 in 2022; margin was 27.5%"


def _subtract(a: str, b: str) -> Program:
    args = [NumberArg(value=Decimal(a)), NumberArg(value=Decimal(b))]
    return Program(steps=[Step(op=Op.SUBTRACT, args=args)])


def test_accepts_grounded_valid_answer() -> None:
    result = SolveResult(ok=True, answer=Decimal("94"), program=_subtract("5829", "5735"))
    verdict = validate(result, _EVIDENCE)
    assert verdict.accept
    assert verdict.total_operands == 2
    assert verdict.grounded_operands == 2


def test_abstains_on_fabricated_operand() -> None:
    result = SolveResult(ok=True, answer=Decimal("4264"), program=_subtract("9999", "5735"))
    verdict = validate(result, _EVIDENCE)
    assert not verdict.accept
    assert any("ungrounded" in reason for reason in verdict.abstain_reasons)


def test_abstains_on_invalid_program() -> None:
    result = SolveResult(ok=False, error="exec_error:div_by_zero")
    verdict = validate(result, _EVIDENCE)
    assert not verdict.accept
    assert any("program_invalid" in reason for reason in verdict.abstain_reasons)


def test_abstains_on_insane_magnitude() -> None:
    args = [NumberArg(value=Decimal("100")), NumberArg(value=Decimal("100"))]
    program = Program(steps=[Step(op=Op.MULTIPLY, args=args)])
    result = SolveResult(ok=True, answer=Decimal("1e20"), program=program)
    verdict = validate(result, "values 100 and 100 appear here")
    assert not verdict.accept
    assert "numeric_insane" in verdict.abstain_reasons


def test_evidence_numbers_parses_formats() -> None:
    numbers = evidence_numbers("revenue $5,829 (1,000) 27.5% and 42")
    assert {Decimal("5829"), Decimal("-1000"), Decimal("27.5"), Decimal("42")} <= numbers


def test_is_grounded_constant_evidence_and_xbrl() -> None:
    assert is_grounded(Decimal("100"), set(), None)  # math constant
    assert is_grounded(Decimal("5829"), {Decimal("5829")}, None)  # in evidence
    assert not is_grounded(Decimal("777"), {Decimal("5829")}, None)  # fabricated
    fact = XbrlFact(
        taxonomy="us-gaap",
        concept="Revenues",
        unit="USD",
        value=Decimal("383285000000"),
        period_end="2023-09-30",
        fiscal_year=2023,
        fiscal_period="FY",
        form="10-K",
    )
    assert is_grounded(Decimal("383285000000"), set(), [fact])
