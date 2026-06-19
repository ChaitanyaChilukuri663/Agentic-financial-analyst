"""Determinism baseline (P2): LLM-proposes-program+executor vs LLM-computes-directly.

Both modes run over FinQA examples; we report answer accuracy for each, split by gold
program complexity (single-step vs multi-step). The headline isn't just the overall
lift — it's that the determinism advantage grows with arithmetic complexity, because a
strong model rarely fumbles a one-step ratio but does fumble chained calculations.

Requires a configured LLM provider (costs a few cents). Not part of the mocked test
suite. Usage:

    python -m ledgerlens.evaluation.finqa_qa data/finqa/dev.json 100
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from ledgerlens.evaluation.finqa import FinqaExample, load_finqa
from ledgerlens.executor.finqa import finqa_cell
from ledgerlens.executor.numbers import answers_match
from ledgerlens.llm import LLMClient, LLMError
from ledgerlens.solver.solver import solve_direct, solve_program

_OP_TOKEN = re.compile(r"[a-z_]+\(")


def _op_count(program: str) -> int:
    return len(_OP_TOKEN.findall(program.lower()))


@dataclass
class BaselineReport:
    """Accuracy + failure tallies for the two answering modes, split by complexity."""

    total: int = 0
    program_correct: int = 0
    direct_correct: int = 0
    program_parse_fail: int = 0
    program_exec_fail: int = 0
    single_total: int = 0
    single_program: int = 0
    single_direct: int = 0
    multi_total: int = 0
    multi_program: int = 0
    multi_direct: int = 0

    @property
    def program_accuracy(self) -> float:
        return self.program_correct / self.total if self.total else 0.0

    @property
    def direct_accuracy(self) -> float:
        return self.direct_correct / self.total if self.total else 0.0


def run_determinism_baseline(
    client: LLMClient, examples: list[FinqaExample], *, limit: int | None = None
) -> BaselineReport:
    """Run both modes over (a prefix of) the examples and tally accuracy by complexity."""
    report = BaselineReport()
    subset = examples[:limit] if limit else examples
    for ex in subset:
        report.total += 1
        context = ex.context()
        result = solve_program(client, ex.question, context, table=ex.table, parse_cell=finqa_cell)
        program_ok = False
        if not result.ok:
            if result.error and result.error.startswith("parse"):
                report.program_parse_fail += 1
            else:
                report.program_exec_fail += 1
        elif answers_match(result.answer, ex.gold_answer):
            program_ok = True
            report.program_correct += 1
        try:
            direct = solve_direct(client, ex.question, context)
        except LLMError:
            direct = ""
        direct_ok = answers_match(direct, ex.gold_answer)
        if direct_ok:
            report.direct_correct += 1
        if _op_count(ex.program) <= 1:
            report.single_total += 1
            report.single_program += int(program_ok)
            report.single_direct += int(direct_ok)
        else:
            report.multi_total += 1
            report.multi_program += int(program_ok)
            report.multi_direct += int(direct_ok)
    return report


def _pct(num: int, den: int) -> str:
    return f"{num / den:.1%}" if den else "n/a"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.finqa_qa <finqa.json> [limit]")
        return 2
    path = Path(argv[0])
    limit = int(argv[1]) if len(argv) > 1 else None
    report = run_determinism_baseline(LLMClient(), load_finqa(path), limit=limit)
    print(f"Determinism baseline: {path}  (n={report.total})")
    print(f"  program + executor: {report.program_correct}  ({report.program_accuracy:.1%})")
    print(f"  LLM direct:         {report.direct_correct}  ({report.direct_accuracy:.1%})")
    print(f"  parse fails: {report.program_parse_fail}   exec fails: {report.program_exec_fail}")
    print(f"  single-step (n={report.single_total}): "
          f"program {_pct(report.single_program, report.single_total)}  "
          f"direct {_pct(report.single_direct, report.single_total)}")
    print(f"  multi-step  (n={report.multi_total}): "
          f"program {_pct(report.multi_program, report.multi_total)}  "
          f"direct {_pct(report.multi_direct, report.multi_total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
