"""Agent eval (P7): accuracy, faithfulness, and abstention over the bundled demo companies.

The agentic layer had per-run telemetry but no benchmark. This scores the ReAct agent on a
hand-labeled set of questions over the three bundled companies (AAPL / MSFT / NVDA):

- **accuracy**  — does the final answer state the gold value / verdict (within tolerance)?
- **faithfulness** — does EVERY number in the answer trace to a tool-verified figure? This is
  the "can't fabricate its numbers" claim *measured*, not assumed: a number in the prose that
  no tool returned counts as ungrounded.
- **abstention** — on questions whose data isn't in the filing, does the agent decline rather
  than invent a figure?

Gold values are the real filed figures from the committed bundles. Requires a configured LLM
provider (costs a few cents); not part of the mocked test suite. Usage:

    python -m ledgerlens.evaluation.agent_eval
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ledgerlens.agent import ResearchAgent, WorkspaceRegistry, make_dispatch
from ledgerlens.agent.agent import Report
from ledgerlens.ingest.edgar import EdgarClient
from ledgerlens.llm import LLMClient

_TOL = Decimal("0.02")  # relative tolerance; absorbs "391.0 billion" vs 391,035,000,000
_UA = "Agentic Financial Analyst eval (contact@example.com)"


@dataclass(frozen=True)
class EvalCase:
    """One labelled question. ``gold`` is a value (lookup) or a ratio (growth); ``winner`` is
    the expected verdict token(s) for a comparison; ``abstain`` marks unanswerable questions."""

    question: str
    category: str  # lookup | growth | compare | abstain
    gold: Decimal | None = None
    winner: tuple[str, ...] = ()
    abstain: bool = False


# Gold = real filed figures from the committed bundles (verified against SEC).
CASES: list[EvalCase] = [
    # --- latest-value lookups ---
    EvalCase("What was Apple's (AAPL) revenue in its most recent fiscal year?",
             "lookup", Decimal("416161000000")),
    EvalCase("What was Microsoft's (MSFT) revenue in its most recent fiscal year?",
             "lookup", Decimal("281724000000")),
    EvalCase("What was NVIDIA's (NVDA) revenue in its most recent fiscal year?",
             "lookup", Decimal("215938000000")),
    EvalCase("What was Apple's (AAPL) net income in its most recent fiscal year?",
             "lookup", Decimal("112010000000")),
    EvalCase("What was Microsoft's (MSFT) net income in its most recent fiscal year?",
             "lookup", Decimal("101832000000")),
    EvalCase("What was NVIDIA's (NVDA) net income in its most recent fiscal year?",
             "lookup", Decimal("120067000000")),
    EvalCase("What was Apple's (AAPL) gross profit in its most recent fiscal year?",
             "lookup", Decimal("195201000000")),
    # --- specific prior-year lookups (tests it doesn't just grab 'latest') ---
    EvalCase("What was Apple's (AAPL) revenue in fiscal year 2023?",
             "lookup", Decimal("383285000000")),
    EvalCase("What was Microsoft's (MSFT) revenue in fiscal year 2024?",
             "lookup", Decimal("245122000000")),
    # --- most-recent-year growth rates ---
    EvalCase("How fast did Apple's (AAPL) revenue grow in its most recent fiscal year?",
             "growth", Decimal("0.06425")),
    EvalCase("How fast did Microsoft's (MSFT) revenue grow in its most recent fiscal year?",
             "growth", Decimal("0.14932")),
    EvalCase("How fast did NVIDIA's (NVDA) revenue grow in its most recent fiscal year?",
             "growth", Decimal("0.65474")),
    EvalCase("How fast did Apple's (AAPL) net income grow in its most recent fiscal year?",
             "growth", Decimal("0.19495")),
    EvalCase("How fast did Microsoft's (MSFT) net income grow in its most recent fiscal year?",
             "growth", Decimal("0.15540")),
    EvalCase("How fast did NVIDIA's (NVDA) net income grow in its most recent fiscal year?",
             "growth", Decimal("0.64746")),
    EvalCase("How fast did Microsoft's (MSFT) gross profit grow in its most recent year?",
             "growth", Decimal("0.13382")),
    # --- cross-company comparisons ---
    EvalCase("Which grew revenue faster last fiscal year, Apple (AAPL) or Microsoft (MSFT)?",
             "compare", winner=("microsoft", "msft")),
    EvalCase("Which grew revenue faster last fiscal year, Apple (AAPL) or NVIDIA (NVDA)?",
             "compare", winner=("nvidia", "nvda")),
    EvalCase("Which grew revenue faster last fiscal year, Microsoft (MSFT) or NVIDIA (NVDA)?",
             "compare", winner=("nvidia", "nvda")),
    # --- abstention: the figure isn't in the filing's XBRL for the demo bundle ---
    EvalCase("What was Apple's (AAPL) inventory in its most recent fiscal year?",
             "abstain", abstain=True),
    EvalCase("What is Microsoft's (MSFT) total employee headcount?",
             "abstain", abstain=True),
    EvalCase("What was NVIDIA's (NVDA) advertising expense last fiscal year?",
             "abstain", abstain=True),
    EvalCase("What is Apple's (AAPL) customer satisfaction score?",
             "abstain", abstain=True),
]

_NUM_RE = re.compile(
    r"\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|percent|%)?",
    re.IGNORECASE,
)
_MULT = {
    "trillion": Decimal("1000000000000"),
    "billion": Decimal("1000000000"),
    "million": Decimal("1000000"),
    "thousand": Decimal("1000"),
}
_ABSTAIN_HINTS = ("not", "no ", "couldn't", "could not", "unable", "n/a", "cannot", "don't have")


def _extract_numbers(text: str) -> list[Decimal]:
    """Pull numeric magnitudes from prose, normalising units and skipping bare fiscal years."""
    out: list[Decimal] = []
    for match in _NUM_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        unit = (match.group(2) or "").lower()
        if unit in ("percent", "%"):
            out.append(value / Decimal(100))
        elif unit in _MULT:
            out.append(value * _MULT[unit])
        elif raw.lstrip("-").isdigit() and 1900 <= abs(int(raw)) <= 2100:
            continue  # bare 4-digit number is almost certainly a fiscal year, not a value
        else:
            out.append(value)
    return out


def _verified_numbers(report: Report) -> list[Decimal]:
    """The figures the agent actually verified — leading value of each successful tool step."""
    nums: list[Decimal] = []
    for step in report.steps:
        if not step.ok:
            continue
        token = step.observation.split()[0].replace(",", "") if step.observation else ""
        try:
            nums.append(Decimal(token))
        except InvalidOperation:
            continue
    return nums


def _matches(a: Decimal, b: Decimal, tol: Decimal = _TOL) -> bool:
    scale = max(abs(a), abs(b))
    return abs(a - b) <= tol * scale if scale else a == b


def _grounded(number: Decimal, verified: list[Decimal]) -> bool:
    return any(_matches(number, v) for v in verified)


@dataclass
class CaseResult:
    case: EvalCase
    summary: str
    correct: bool
    faithful: bool
    summary_numbers: int
    grounded_numbers: int


@dataclass
class AgentEvalReport:
    results: list[CaseResult] = field(default_factory=list)

    def _by(self, category: str) -> list[CaseResult]:
        return [r for r in self.results if r.case.category == category]

    def accuracy(self, category: str) -> tuple[int, int]:
        rows = self._by(category)
        return sum(r.correct for r in rows), len(rows)

    @property
    def answerable_correct(self) -> tuple[int, int]:
        rows = [r for r in self.results if not r.case.abstain]
        return sum(r.correct for r in rows), len(rows)

    @property
    def faithfulness_numbers(self) -> tuple[int, int]:
        return (
            sum(r.grounded_numbers for r in self.results),
            sum(r.summary_numbers for r in self.results),
        )


def run_agent_eval(client: LLMClient, cases: list[EvalCase]) -> AgentEvalReport:
    registry = WorkspaceRegistry(EdgarClient(_UA))
    report = AgentEvalReport()
    for case in cases:
        agent = ResearchAgent(client, make_dispatch(registry), max_steps=8)
        run = agent.run(case.question)
        verified = _verified_numbers(run)
        summary_nums = _extract_numbers(run.summary)
        grounded = [n for n in summary_nums if _grounded(n, verified)]
        faithful = len(grounded) == len(summary_nums)

        if case.abstain:
            # Correct = did not assert an (ungrounded) figure; ideally signals it couldn't find it.
            said_no = any(hint in run.summary.lower() for hint in _ABSTAIN_HINTS)
            correct = faithful and (said_no or not summary_nums)
        elif case.category == "compare":
            correct = any(w in run.summary.lower() for w in case.winner)
        else:
            correct = case.gold is not None and any(
                _matches(n, case.gold) for n in summary_nums
            )
        report.results.append(
            CaseResult(case, run.summary, correct, faithful, len(summary_nums), len(grounded))
        )
    return report


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "n/a"


def _render(report: AgentEvalReport) -> str:
    lines = ["# Agent eval\n"]
    ac, at = report.answerable_correct
    gn, tn = report.faithfulness_numbers
    lines.append(f"- **Answer accuracy (answerable):** {ac}/{at} ({_pct(ac, at)})")
    for cat in ("lookup", "growth", "compare"):
        c, t = report.accuracy(cat)
        lines.append(f"  - {cat}: {c}/{t} ({_pct(c, t)})")
    absc, abst = report.accuracy("abstain")
    lines.append(f"- **Abstention correctness:** {absc}/{abst} ({_pct(absc, abst)})")
    lines.append(
        f"- **Faithfulness:** {gn}/{tn} numbers in answers trace to a verified figure "
        f"({_pct(gn, tn)})"
    )
    lines.append("\n| # | category | question | correct | faithful | answer |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(report.results, 1):
        q = r.case.question if len(r.case.question) < 60 else r.case.question[:57] + "..."
        ans = r.summary if len(r.summary) < 70 else r.summary[:67] + "..."
        lines.append(
            f"| {i} | {r.case.category} | {q} | {'✅' if r.correct else '❌'} | "
            f"{'✅' if r.faithful else '⚠️'} | {ans} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252 and choke on the ✅/⚠️ glyphs in the report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    out_path = Path(argv[0]) if argv else Path("evals/agent_results.md")
    report = run_agent_eval(LLMClient(), CASES)
    rendered = _render(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")  # write before printing so a print error
    print(rendered)                                  # can never lose the (LLM-costly) results
    print(f"(written to {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
