"""Read SEC XBRL companyfacts into a flat list of facts.

XBRL gives machine-readable, unit-tagged, period-stamped numeric facts — a
ground-truth anchor for operand grounding (P4): does a number the LLM extracted
actually match a value the company filed?
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_REL_TOL = Decimal("0.001")


@dataclass(frozen=True)
class XbrlFact:
    """One reported numeric fact."""

    taxonomy: str
    concept: str
    unit: str
    value: Decimal
    period_end: str
    fiscal_year: int | None
    fiscal_period: str
    form: str


def parse_company_facts(payload: dict) -> list[XbrlFact]:
    """Flatten a companyfacts JSON payload into a list of :class:`XbrlFact`."""
    facts: list[XbrlFact] = []
    for taxonomy, concepts in payload.get("facts", {}).items():
        for concept, body in concepts.items():
            for unit, entries in body.get("units", {}).items():
                for entry in entries:
                    value = _to_decimal(entry.get("val"))
                    if value is None:
                        continue
                    facts.append(
                        XbrlFact(
                            taxonomy=taxonomy,
                            concept=concept,
                            unit=unit,
                            value=value,
                            period_end=entry.get("end", ""),
                            fiscal_year=entry.get("fy"),
                            fiscal_period=entry.get("fp", ""),
                            form=entry.get("form", ""),
                        )
                    )
    return facts


def match_value(
    facts: list[XbrlFact], value: Decimal, *, rel_tol: Decimal = _REL_TOL
) -> list[XbrlFact]:
    """Return facts whose value matches ``value`` within relative tolerance."""
    matches: list[XbrlFact] = []
    for fact in facts:
        scale = max(abs(fact.value), abs(value))
        if scale == 0:
            if fact.value == value:
                matches.append(fact)
        elif abs(fact.value - value) <= rel_tol * scale:
            matches.append(fact)
    return matches


def _to_decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None
