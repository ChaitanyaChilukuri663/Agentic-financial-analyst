"""The deterministic executor.

Pure and *total*: it never raises on a malformed program — it returns an
:class:`ExecResult` carrying either the value or a structured :class:`ProgramError`.
That error channel is what the abstention/validation gates consume later (P4).
The LLM never performs arithmetic; this module is the only thing that computes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation

from ledgerlens.executor.dsl import (
    MATH_OPS,
    TABLE_OPS,
    Arg,
    NumberArg,
    Op,
    Program,
    RefArg,
    RowArg,
    Step,
    Table,
)
from ledgerlens.executor.numbers import str_to_num

Value = Decimal | bool


@dataclass(frozen=True)
class ProgramError:
    """A structured execution failure."""

    code: str
    message: str
    step_index: int | None = None


@dataclass
class ExecResult:
    """The outcome of executing a program: a value, or an error."""

    ok: bool
    value: Value | None = None
    steps: list[Value] = field(default_factory=list)
    error: ProgramError | None = None


def execute(
    program: Program,
    table: Table | None = None,
    *,
    parse_cell: Callable[[str], Decimal | None] = str_to_num,
) -> ExecResult:
    """Execute ``program`` against an optional ``table`` and return an ExecResult.

    ``parse_cell`` controls how table cells are turned into numbers; the default is
    the robust LedgerLens parser. The FinQA replay injects a FinQA-faithful parser.
    """
    results: list[Value] = []
    for i, step in enumerate(program.steps):
        outcome = _run_step(step, results, table, i, parse_cell)
        if isinstance(outcome, ProgramError):
            return ExecResult(ok=False, steps=results, error=outcome)
        results.append(outcome)
    if not results:
        return ExecResult(ok=False, error=ProgramError("empty", "program has no steps"))
    return ExecResult(ok=True, value=results[-1], steps=results)


def _run_step(
    step: Step,
    results: list[Value],
    table: Table | None,
    idx: int,
    parse_cell: Callable[[str], Decimal | None],
) -> Value | ProgramError:
    if step.op in MATH_OPS:
        return _run_math(step, results, idx)
    if step.op in TABLE_OPS:
        return _run_table(step, table, idx, parse_cell)
    return ProgramError("unknown_op", f"unsupported op {step.op}", idx)


def _run_math(step: Step, results: list[Value], idx: int) -> Value | ProgramError:
    if len(step.args) != 2:
        return ProgramError("arity", f"{step.op} expects 2 args, got {len(step.args)}", idx)
    a = _resolve_number(step.args[0], results, idx)
    if isinstance(a, ProgramError):
        return a
    b = _resolve_number(step.args[1], results, idx)
    if isinstance(b, ProgramError):
        return b
    try:
        match step.op:
            case Op.ADD:
                return a + b
            case Op.SUBTRACT:
                return a - b
            case Op.MULTIPLY:
                return a * b
            case Op.DIVIDE:
                if b == 0:
                    return ProgramError("div_by_zero", "division by zero", idx)
                return a / b
            case Op.EXP:
                return a**b
            case Op.GREATER:
                return a > b
    except (InvalidOperation, DivisionByZero, OverflowError, ValueError) as exc:
        return ProgramError("math_error", f"{step.op}: {exc}", idx)
    return ProgramError("unknown_op", f"unhandled {step.op}", idx)


def _run_table(
    step: Step,
    table: Table | None,
    idx: int,
    parse_cell: Callable[[str], Decimal | None],
) -> Value | ProgramError:
    if table is None:
        return ProgramError("no_table", f"{step.op} requires a table", idx)
    if not step.args or not isinstance(step.args[0], RowArg):
        return ProgramError("type", f"{step.op} expects a row name as its first arg", idx)
    row = table.find_row(step.args[0].name)
    if row is None:
        return ProgramError("row_not_found", f"row {step.args[0].name!r} not found", idx)
    nums = [n for n in (parse_cell(c) for c in row[1:]) if n is not None]
    if not nums:
        return ProgramError("empty_row", f"row {step.args[0].name!r} has no numeric cells", idx)
    match step.op:
        case Op.TABLE_SUM:
            return sum(nums, Decimal(0))
        case Op.TABLE_AVERAGE:
            return sum(nums, Decimal(0)) / Decimal(len(nums))
        case Op.TABLE_MAX:
            return max(nums)
        case Op.TABLE_MIN:
            return min(nums)
    return ProgramError("unknown_op", f"unhandled {step.op}", idx)


def _resolve_number(arg: Arg, results: list[Value], idx: int) -> Decimal | ProgramError:
    if isinstance(arg, NumberArg):
        return arg.value
    if isinstance(arg, RefArg):
        if arg.step < 0 or arg.step >= len(results):
            return ProgramError("bad_ref", f"#{arg.step} out of range", idx)
        value = results[arg.step]
        if isinstance(value, bool):
            return ProgramError("type", f"#{arg.step} is a boolean, not a number", idx)
        return value
    return ProgramError("type", f"{arg.kind} arg is not a number", idx)
