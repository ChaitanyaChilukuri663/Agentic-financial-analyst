"""Deterministic program-DSL executor (P1) — the crown jewel.

Runs the symbolic reasoning program proposed by the LLM (operands + ops). Pure
Python, total (returns a value-or-error result), and unit-agnostic. The LLM never
performs arithmetic here.
"""

from ledgerlens.executor.dsl import (
    Arg,
    NoneArg,
    NumberArg,
    Op,
    Program,
    RefArg,
    RowArg,
    Step,
    Table,
)
from ledgerlens.executor.finqa import FinqaProgramError, finqa_cell, parse_finqa_program
from ledgerlens.executor.numbers import answers_match, str_to_num
from ledgerlens.executor.program import ExecResult, ProgramError, execute

__all__ = [
    "Arg",
    "ExecResult",
    "FinqaProgramError",
    "NoneArg",
    "NumberArg",
    "Op",
    "Program",
    "ProgramError",
    "RefArg",
    "RowArg",
    "Step",
    "Table",
    "answers_match",
    "execute",
    "finqa_cell",
    "parse_finqa_program",
    "str_to_num",
]
