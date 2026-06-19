"""Prompt construction for the extract+plan and direct-answer modes."""

from __future__ import annotations

_PROGRAM_SYSTEM = """You are a financial-analysis planner. Given evidence from a filing and a
question, you do NOT compute the answer. You output a symbolic reasoning PROGRAM that a
separate deterministic engine executes.

Program syntax — operations joined by ", ":
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)  greater(a, b)
  table_sum(row, none)  table_average(row, none)  table_max(row, none)  table_min(row, none)

Operands are numbers taken verbatim from the evidence (e.g. 5829), constants such as
const_100 / const_1000000 / const_m1, table row names, or references to earlier steps
written as #0, #1, ... (0-indexed). Write percentages with their % sign (e.g. 12.5%); the
engine divides them by 100. The last step's result is the answer. Never put a computed
final number in the program — only operations over operands.

Example
Evidence: net revenue was $5,829 in 2023 and $5,735 in 2022.
Question: what was the percent change in net revenue from 2022 to 2023?
program: subtract(5829, 5735), divide(#0, 5735)"""

_DIRECT_SYSTEM = """You are a financial analyst. Given evidence from a filing and a question,
compute and return the final answer — a number (e.g. 0.0164 or 14.1%) or 'yes'/'no'. Return
the answer and brief reasoning."""


def build_program_messages(question: str, context: str) -> list[dict[str, str]]:
    """Messages that ask the model to propose a reasoning program (not the answer)."""
    return [
        {"role": "system", "content": _PROGRAM_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"},
    ]


def build_direct_messages(question: str, context: str) -> list[dict[str, str]]:
    """Messages for the baseline where the model computes the answer directly."""
    return [
        {"role": "system", "content": _DIRECT_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {question}"},
    ]
