"""Tests for parsing FinQA gold program strings, plus an end-to-end replay smoke test."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ledgerlens.executor.dsl import NoneArg, NumberArg, Op, RefArg, RowArg
from ledgerlens.executor.finqa import FinqaProgramError, finqa_cell, parse_finqa_program
from ledgerlens.executor.numbers import answers_match
from ledgerlens.executor.program import execute


def test_parse_two_step_program() -> None:
    program = parse_finqa_program("subtract(5829, 5735), divide(#0, 5735)")
    assert [s.op for s in program.steps] == [Op.SUBTRACT, Op.DIVIDE]
    first = program.steps[0].args[0]
    assert isinstance(first, NumberArg)
    assert first.value == Decimal("5829")
    ref = program.steps[1].args[0]
    assert isinstance(ref, RefArg)
    assert ref.step == 0


def test_parse_const_row_and_none() -> None:
    program = parse_finqa_program("table_sum(net income, none), divide(#0, const_1000000)")
    row = program.steps[0].args[0]
    assert isinstance(row, RowArg)
    assert row.name == "net income"
    assert isinstance(program.steps[0].args[1], NoneArg)
    constant = program.steps[1].args[1]
    assert isinstance(constant, NumberArg)
    assert constant.value == Decimal("1000000")


def test_parse_rejects_unknown_op() -> None:
    with pytest.raises(FinqaProgramError):
        parse_finqa_program("frobnicate(1, 2)")


def test_parse_rejects_empty() -> None:
    with pytest.raises(FinqaProgramError):
        parse_finqa_program("")


def test_end_to_end_parse_execute_compare() -> None:
    program = parse_finqa_program("subtract(5829, 5735), divide(#0, 5735)")
    result = execute(program)
    assert result.ok
    assert answers_match(result.value, 0.016390)


def test_finqa_cell_cleaning() -> None:
    assert finqa_cell("$ -6129 ( 6129 )") == Decimal("-6129")
    assert finqa_cell("11.4% ( 11.4 % )") == Decimal("0.114")
    assert finqa_cell("-13 ( 13 )") == Decimal("-13")
