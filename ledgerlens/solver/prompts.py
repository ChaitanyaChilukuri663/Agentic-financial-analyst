"""Prompt construction for the extract+plan and direct-answer modes."""

from __future__ import annotations

_PROGRAM_SYSTEM = """You are a financial-analysis planner. Given evidence from a filing and a
question, you do NOT compute the answer. You output a symbolic reasoning PROGRAM that a
separate deterministic engine executes.

Operations (join multiple with ", "):
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)  greater(a, b)
  table_sum(row, none)  table_average(row, none)  table_max(row, none)  table_min(row, none)

Rules:
- Operands are numbers copied verbatim from the evidence, WITHOUT $ or % signs — write
  27.5, never $27.5 or 27.5%. Do not pre-scale percentages.
- Constants are written const_100, const_1000, const_1000000, const_m1, etc.
- Reference an earlier step's result as #0, #1, ... (0-indexed); the last step is the answer.
- table_sum / table_average / table_max / table_min take exactly ONE table ROW NAME (a row
  label that appears in the evidence), never a list of numbers. To combine specific numbers,
  use add / subtract / multiply / divide.
- Always return a non-empty program, and never put a final computed number in it.

Examples
Evidence: net revenue was $5,829 in 2023 and $5,735 in 2022.
Question: what was the percent change in net revenue from 2022 to 2023?
program: subtract(5829, 5735), divide(#0, 5735)

Evidence: residential mortgages were 1356 in 2013 and 2220 in 2012.
Question: total residential mortgages for 2013 and 2012?
program: add(1356, 2220)

Evidence: gross margin was 27.5% in fiscal 2004 and 27.3% in fiscal 2003.
Question: gross margin decline in fiscal 2004 from 2003?
program: subtract(27.5, 27.3)"""

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
