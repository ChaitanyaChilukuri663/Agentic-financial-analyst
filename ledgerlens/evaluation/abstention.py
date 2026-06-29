"""Abstention eval (P4): do the gates buy precision, and what do the misses look like?

For each FinQA example: solve_program -> validate (gates) -> record answered/correct, and
for every wrong answer classify the failure against the gold program. The story: when the
gates let an answer through, it should be right more often than the unfiltered model — the
system trades a little coverage for higher precision. Requires an LLM provider; not part of
the mocked test suite.

    python -m ledgerlens.evaluation.abstention data/finqa/dev.json 100
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ledgerlens.evaluation.finqa import FinqaExample, load_finqa
from ledgerlens.executor.finqa import finqa_cell
from ledgerlens.executor.numbers import answers_match
from ledgerlens.llm import LLMClient, LLMError
from ledgerlens.solver.solver import SolveResult, solve_program
from ledgerlens.validation import validate

_OP_TOKEN = re.compile(r"[a-z_]+\(")


def _ops(program: str) -> list[str]:
    return _OP_TOKEN.findall(program.lower())


@dataclass
class AbstentionReport:
    """Coverage / precision tallies plus an error taxonomy over wrong answers."""

    total: int = 0
    overall_correct: int = 0
    answered: int = 0
    answered_correct: int = 0
    abstained: int = 0
    abstained_correct: int = 0
    taxonomy: Counter[str] = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def overall_accuracy(self) -> float:
        return self.overall_correct / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.answered_correct / self.answered if self.answered else 0.0


def _classify(result: SolveResult, gold_program: str) -> str:
    if not result.ok or result.proposal is None:
        return "parse_or_exec_fail"
    if _ops(result.proposal.program) != _ops(gold_program):
        return "wrong_program_structure"
    return "wrong_operands"


def run_abstention_eval(
    client: LLMClient, examples: list[FinqaExample], *, limit: int | None = None
) -> AbstentionReport:
    """Solve, gate, and tally each example."""
    report = AbstentionReport()
    subset = examples[:limit] if limit else examples
    for ex in subset:
        report.total += 1
        context = ex.context()
        try:
            result = solve_program(
                client, ex.question, context, table=ex.table, parse_cell=finqa_cell
            )
        except LLMError:
            result = SolveResult(ok=False, error="llm_error")
        correct = result.ok and answers_match(result.answer, ex.gold_answer)
        if correct:
            report.overall_correct += 1
        if validate(result, context).accept:
            report.answered += 1
            report.answered_correct += int(correct)
        else:
            report.abstained += 1
            report.abstained_correct += int(correct)
        if not correct:
            report.taxonomy[_classify(result, ex.program)] += 1
    return report


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.abstention <finqa.json> [limit]")
        return 2
    path = Path(argv[0])
    limit = int(argv[1]) if len(argv) > 1 else None
    report = run_abstention_eval(LLMClient(), load_finqa(path), limit=limit)
    false_abstain = report.abstained_correct / report.abstained if report.abstained else 0.0
    print(f"Abstention eval: {path}  (n={report.total})")
    print(f"  overall accuracy (no gates): {report.overall_accuracy:.1%}")
    print(
        f"  coverage (answered):         {report.coverage:.1%}  "
        f"({report.answered}/{report.total})"
    )
    print(f"  precision on answered:       {report.precision:.1%}")
    print(f"  false-abstain rate:          {false_abstain:.1%}  (correct ones declined)")
    print("  error taxonomy (of wrong answers):")
    for label, count in report.taxonomy.most_common():
        print(f"    {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
