"""Load FinQA examples and replay their gold programs through the executor.

This validates the executor independently of any LLM: if it reproduces the gold
``exe_ans`` for (almost) every gold program, the executor is correct. The few it
misses are diagnostic — they expose DSL/number-parsing gaps.

FinQA dataset: Chen et al., 2021 (https://github.com/czyssrs/FinQA), used under its
license. No dataset content is committed to this repo.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ledgerlens.executor.dsl import Table
from ledgerlens.executor.finqa import FinqaProgramError, finqa_cell, parse_finqa_program
from ledgerlens.executor.numbers import answers_match
from ledgerlens.executor.program import execute


@dataclass
class FinqaExample:
    """A single FinQA item: the gold program/answer plus the question and context."""

    uid: str
    program: str
    gold_answer: object
    table: Table
    question: str = ""
    pre_text: str = ""
    post_text: str = ""

    def context(self) -> str:
        """Assemble the evidence the solver sees: pre-text, table rows, post-text."""
        table_text = "\n".join(" | ".join(row) for row in self.table.rows)
        parts = [self.pre_text.strip(), table_text.strip(), self.post_text.strip()]
        return "\n\n".join(part for part in parts if part)


@dataclass
class ReplayReport:
    """Aggregate outcome of replaying a set of gold programs."""

    total: int = 0
    reproduced: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    mismatches: list[tuple[str, object, object]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.reproduced / self.total if self.total else 0.0


def load_finqa(path: str | Path) -> list[FinqaExample]:
    """Load FinQA examples (with a gold program) from a FinQA JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[FinqaExample] = []
    for item in data:
        qa = item.get("qa") or {}
        program = qa.get("program")
        if not program:
            continue
        pre = item.get("pre_text", [])
        post = item.get("post_text", [])
        examples.append(
            FinqaExample(
                uid=item.get("id", ""),
                program=program,
                gold_answer=qa.get("exe_ans"),
                table=Table(rows=[[str(cell) for cell in row] for row in item.get("table", [])]),
                question=qa.get("question", ""),
                pre_text=" ".join(pre) if isinstance(pre, list) else str(pre),
                post_text=" ".join(post) if isinstance(post, list) else str(post),
            )
        )
    return examples


def replay_finqa(examples: list[FinqaExample], *, max_mismatches: int = 20) -> ReplayReport:
    """Parse + execute each gold program and compare to the gold answer."""
    report = ReplayReport()
    for ex in examples:
        report.total += 1
        try:
            program = parse_finqa_program(ex.program)
        except FinqaProgramError:
            report.outcomes["parse_error"] += 1
            continue
        result = execute(program, ex.table, parse_cell=finqa_cell)
        if not result.ok and result.error is not None:
            report.outcomes[f"exec_error:{result.error.code}"] += 1
            continue
        if answers_match(result.value, ex.gold_answer):
            report.reproduced += 1
            report.outcomes["reproduced"] += 1
        else:
            report.outcomes["wrong_answer"] += 1
            if len(report.mismatches) < max_mismatches:
                report.mismatches.append((ex.uid, result.value, ex.gold_answer))
    return report
