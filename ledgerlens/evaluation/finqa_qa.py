"""Determinism baseline (P2): LLM-proposes-program+executor vs LLM-computes-directly.

Both modes run over FinQA examples; we report answer accuracy for each. This is the
first "money chart" — the determinism lift from routing arithmetic through the executor
instead of letting the model do mental math.

Requires a configured LLM provider (costs a few cents). Not part of the mocked test
suite. Usage:

    python -m ledgerlens.evaluation.finqa_qa data/finqa/dev.json 100
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ledgerlens.evaluation.finqa import FinqaExample, load_finqa
from ledgerlens.executor.finqa import finqa_cell
from ledgerlens.executor.numbers import answers_match
from ledgerlens.llm import LLMClient, LLMError
from ledgerlens.solver.solver import solve_direct, solve_program


@dataclass
class BaselineReport:
    """Accuracy + failure tallies for the two answering modes."""

    total: int = 0
    program_correct: int = 0
    direct_correct: int = 0
    program_parse_fail: int = 0
    program_exec_fail: int = 0

    @property
    def program_accuracy(self) -> float:
        return self.program_correct / self.total if self.total else 0.0

    @property
    def direct_accuracy(self) -> float:
        return self.direct_correct / self.total if self.total else 0.0


def run_determinism_baseline(
    client: LLMClient, examples: list[FinqaExample], *, limit: int | None = None
) -> BaselineReport:
    """Run both modes over (a prefix of) the examples and tally accuracy."""
    report = BaselineReport()
    subset = examples[:limit] if limit else examples
    for ex in subset:
        report.total += 1
        context = ex.context()
        result = solve_program(client, ex.question, context, table=ex.table, parse_cell=finqa_cell)
        if not result.ok:
            if result.error and result.error.startswith("parse"):
                report.program_parse_fail += 1
            else:
                report.program_exec_fail += 1
        elif answers_match(result.answer, ex.gold_answer):
            report.program_correct += 1
        try:
            direct = solve_direct(client, ex.question, context)
        except LLMError:
            direct = ""
        if answers_match(direct, ex.gold_answer):
            report.direct_correct += 1
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.finqa_qa <finqa.json> [limit]")
        return 2
    path = Path(argv[0])
    limit = int(argv[1]) if len(argv) > 1 else None
    examples = load_finqa(path)
    report = run_determinism_baseline(LLMClient(), examples, limit=limit)
    print(f"Determinism baseline: {path}  (n={report.total})")
    print(f"  program + executor: {report.program_correct}  ({report.program_accuracy:.1%})")
    print(f"  LLM direct:         {report.direct_correct}  ({report.direct_accuracy:.1%})")
    print(f"  program parse fails: {report.program_parse_fail}")
    print(f"  program exec fails:  {report.program_exec_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
