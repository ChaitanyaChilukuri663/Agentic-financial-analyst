"""Tests for the deterministic executor — every op, references, and error channels."""

from __future__ import annotations

from decimal import Decimal

from ledgerlens.executor.dsl import (
    NoneArg,
    NumberArg,
    Op,
    Program,
    RefArg,
    RowArg,
    Step,
    Table,
)
from ledgerlens.executor.finqa import finqa_cell
from ledgerlens.executor.program import execute


def _num(x: object) -> NumberArg:
    return NumberArg(value=Decimal(str(x)))


def _step(op: Op, *args: object) -> Step:
    return Step(op=op, args=list(args))


def test_subtract_then_divide_chain() -> None:
    program = Program(
        steps=[
            _step(Op.SUBTRACT, _num(5829), _num(5735)),
            _step(Op.DIVIDE, RefArg(step=0), _num(5735)),
        ]
    )
    result = execute(program)
    assert result.ok
    assert abs(result.value - Decimal("0.016390")) < Decimal("0.0001")


def test_add_multiply_exp() -> None:
    assert execute(Program(steps=[_step(Op.ADD, _num(2), _num(3))])).value == Decimal("5")
    assert execute(Program(steps=[_step(Op.MULTIPLY, _num(4), _num(5))])).value == Decimal("20")
    assert execute(Program(steps=[_step(Op.EXP, _num(2), _num(3))])).value == Decimal("8")


def test_greater_returns_bool() -> None:
    assert execute(Program(steps=[_step(Op.GREATER, _num(10), _num(3))])).value is True
    assert execute(Program(steps=[_step(Op.GREATER, _num(1), _num(3))])).value is False


def test_divide_by_zero() -> None:
    result = execute(Program(steps=[_step(Op.DIVIDE, _num(1), _num(0))]))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "div_by_zero"


def test_bad_reference() -> None:
    result = execute(Program(steps=[_step(Op.ADD, RefArg(step=5), _num(1))]))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "bad_ref"


def test_arity_error() -> None:
    result = execute(Program(steps=[_step(Op.ADD, _num(1))]))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "arity"


def test_ref_to_boolean_is_type_error() -> None:
    program = Program(
        steps=[
            _step(Op.GREATER, _num(2), _num(1)),
            _step(Op.ADD, RefArg(step=0), _num(1)),
        ]
    )
    result = execute(program)
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "type"


def test_empty_program() -> None:
    result = execute(Program(steps=[]))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "empty"


def _sample_table() -> Table:
    return Table(
        rows=[
            ["revenue", "100", "200", "300"],
            ["cost", "(50)", "n/a", "$25"],
        ]
    )


def test_table_sum_and_average() -> None:
    table = _sample_table()
    total = execute(Program(steps=[_step(Op.TABLE_SUM, RowArg(name="revenue"), NoneArg())]), table)
    avg = execute(
        Program(steps=[_step(Op.TABLE_AVERAGE, RowArg(name="revenue"), NoneArg())]), table
    )
    assert total.value == Decimal("600")
    assert avg.value == Decimal("200")


def test_table_max_and_min_skip_non_numeric() -> None:
    table = _sample_table()
    biggest = execute(Program(steps=[_step(Op.TABLE_MAX, RowArg(name="cost"), NoneArg())]), table)
    smallest = execute(Program(steps=[_step(Op.TABLE_MIN, RowArg(name="cost"), NoneArg())]), table)
    assert biggest.value == Decimal("25")
    assert smallest.value == Decimal("-50")


def test_table_row_not_found() -> None:
    result = execute(
        Program(steps=[_step(Op.TABLE_SUM, RowArg(name="missing"), NoneArg())]),
        _sample_table(),
    )
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "row_not_found"


def test_table_op_without_table() -> None:
    result = execute(Program(steps=[_step(Op.TABLE_SUM, RowArg(name="revenue"), NoneArg())]))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "no_table"


def test_table_op_with_injected_cell_parser() -> None:
    table = Table(rows=[["margin", "11.4% ( 11.4 % )", "7.8% ( 7.8 % )"]])
    result = execute(
        Program(steps=[_step(Op.TABLE_MAX, RowArg(name="margin"), NoneArg())]),
        table,
        parse_cell=finqa_cell,
    )
    assert result.value == Decimal("0.114")
