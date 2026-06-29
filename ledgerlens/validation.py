"""Validation / abstention gates — the "can't fabricate its numbers" guarantee.

A computed answer is only returned if it passes deterministic, *trustworthy* gates:

- **program validity** — the program parsed and executed (no error channel).
- **operand grounding** — every literal operand traces to the evidence (or a filed XBRL
  fact, or is a math constant). This is what catches a fabricated number.
- **numeric sanity** — the result is finite and not absurd in magnitude.

These are signals the LLM cannot fake — not its self-reported confidence. If any gate
fails the answer is withheld (abstain) with reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from ledgerlens.executor.dsl import NumberArg
from ledgerlens.ingest.xbrl import XbrlFact, match_value
from ledgerlens.solver.solver import SolveResult

_NUM_RE = re.compile(r"\(?\$?-?\d[\d,]*\.?\d*\)?%?")
_GROUND_TOL = Decimal("0.001")
_MAX_MAGNITUDE = Decimal("1e15")

# Math constants a program may legitimately use that need not appear in the evidence.
_CONSTANTS = {
    Decimal(x)
    for x in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "100",
              "1000", "10000", "100000", "1000000", "1000000000", "-1", "0.5")
}


@dataclass
class Verdict:
    """The gate decision for one computed answer."""

    accept: bool
    abstain_reasons: list[str] = field(default_factory=list)
    grounded_operands: int = 0
    total_operands: int = 0


def validate(
    result: SolveResult,
    evidence_text: str,
    *,
    xbrl: list[XbrlFact] | None = None,
    min_grounded: float = 1.0,
) -> Verdict:
    """Run the gates over a solve result; accept only if all pass."""
    if not result.ok or result.program is None:
        return Verdict(accept=False, abstain_reasons=[f"program_invalid:{result.error}"])

    reasons: list[str] = []
    if not _numeric_sane(result.answer):
        reasons.append("numeric_insane")

    evidence = evidence_numbers(evidence_text)
    operands = [
        arg.value
        for step in result.program.steps
        for arg in step.args
        if isinstance(arg, NumberArg)
    ]
    grounded = sum(1 for value in operands if is_grounded(value, evidence, xbrl))
    if operands and grounded / len(operands) < min_grounded:
        reasons.append(f"ungrounded_operands:{len(operands) - grounded}/{len(operands)}")

    return Verdict(
        accept=not reasons,
        abstain_reasons=reasons,
        grounded_operands=grounded,
        total_operands=len(operands),
    )


def evidence_numbers(text: str) -> set[Decimal]:
    """Extract the set of numbers present in the evidence (as bare decimals)."""
    numbers: set[Decimal] = set()
    for token in _NUM_RE.findall(text):
        value = _bare_number(token)
        if value is not None:
            numbers.add(value)
    return numbers


def is_grounded(
    value: Decimal,
    evidence: set[Decimal],
    xbrl: list[XbrlFact] | None,
    *,
    tol: Decimal = _GROUND_TOL,
) -> bool:
    """True if ``value`` is a constant, appears in the evidence, or matches an XBRL fact."""
    if value in _CONSTANTS:
        return True
    for candidate in evidence:
        scale = max(abs(candidate), abs(value))
        if scale == 0 or abs(candidate - value) <= tol * scale:
            return True
    return bool(xbrl) and bool(match_value(xbrl, value))


def _numeric_sane(value: Decimal | bool | None) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Decimal):
        return value.is_finite() and abs(value) < _MAX_MAGNITUDE
    return False


def _bare_number(token: str) -> Decimal | None:
    text = token.strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    if not text:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return -value if negative else value
