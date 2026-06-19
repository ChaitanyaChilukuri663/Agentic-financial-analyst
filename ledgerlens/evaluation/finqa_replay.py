"""CLI: replay FinQA gold programs through the executor and print the reproduction rate.

    python -m ledgerlens.evaluation.finqa_replay data/finqa/dev.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from ledgerlens.evaluation.finqa import load_finqa, replay_finqa


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.finqa_replay <finqa.json>")
        return 2
    path = Path(argv[0])
    examples = load_finqa(path)
    report = replay_finqa(examples)
    print(f"FinQA replay: {path}")
    print(f"  examples:   {report.total}")
    print(f"  reproduced: {report.reproduced}  ({report.rate:.1%})")
    for outcome, count in report.outcomes.most_common():
        print(f"    {outcome}: {count}")
    if report.mismatches:
        print("  sample mismatches (uid, computed, gold):")
        for uid, computed, gold in report.mismatches[:10]:
            print(f"    {uid}: {computed} != {gold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
