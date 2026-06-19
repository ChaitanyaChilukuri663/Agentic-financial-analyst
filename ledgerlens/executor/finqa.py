"""Parse FinQA gold program strings into our DSL.

FinQA format: ``subtract(5829, 5735), divide(#0, 5735)`` — comma-joined ``op(args)``
where each arg is a number, a ``const_`` token, a ``#n`` reference, a table row
name, or ``none``. Parsing gold programs lets us validate the executor at scale
(replay) and gives the LLM a worked target format for P2.
"""

from __future__ import annotations

import re
from decimal import Decimal

from ledgerlens.executor.dsl import (
    TABLE_OPS,
    Arg,
    NoneArg,
    NumberArg,
    Op,
    Program,
    RefArg,
    RowArg,
    Step,
)
from ledgerlens.executor.numbers import str_to_num

_OP_RE = re.compile(r"(\w+)\(([^)]*)\)")
_REF_RE = re.compile(r"^#(\d+)$")
_VALID_OPS = {op.value for op in Op}


class FinqaProgramError(ValueError):
    """Raised when a FinQA program string cannot be parsed into the DSL."""


def parse_finqa_program(text: str) -> Program:
    """Parse a FinQA program string into a :class:`Program`."""
    matches = _OP_RE.findall(text or "")
    if not matches:
        raise FinqaProgramError(f"no operations found in program: {text!r}")
    steps: list[Step] = []
    for op_name, arg_blob in matches:
        if op_name not in _VALID_OPS:
            raise FinqaProgramError(f"unsupported op: {op_name!r}")
        op = Op(op_name)
        tokens = [a.strip() for a in arg_blob.split(",") if a.strip()]
        args = [_parse_arg(tok, op=op, pos=pos) for pos, tok in enumerate(tokens)]
        steps.append(Step(op=op, args=args))
    return Program(steps=steps)


def _parse_arg(token: str, *, op: Op, pos: int) -> Arg:
    ref = _REF_RE.match(token)
    if ref:
        return RefArg(step=int(ref.group(1)))
    # In a table op, the first operand is always a row label — even when it looks
    # numeric (e.g. a year like "2016").
    if op in TABLE_OPS and pos == 0:
        return RowArg(name=token)
    if token == "none":
        return NoneArg()
    num = str_to_num(token)
    if num is not None:
        return NumberArg(value=num)
    return RowArg(name=token)


def finqa_cell(text: str) -> Decimal | None:
    """Parse a FinQA table cell, mirroring FinQA's ``process_row`` cleaning.

    FinQA cells look like ``$ -6129 ( 6129 )`` or ``11.4% ( 11.4 % )``: it strips
    ``$``, takes the part before ``(``, then parses (with ``%`` -> ``/100``).
    """
    cleaned = text.replace("$", "").split("(")[0].strip()
    return str_to_num(cleaned)
