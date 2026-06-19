"""FinQA-compatible numeric parsing and answer comparison.

Shared by the executor (parsing table cells / operands) and the eval harness
(comparing computed answers to gold). Kept dependency-free and pure.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

_CONST_PREFIX = "const_"
_REL_TOL = Decimal("0.001")
_ABS_TOL = Decimal("0.000001")


def str_to_num(text: str | None) -> Decimal | None:
    """Parse a financial-text number to ``Decimal``, or ``None`` if not numeric.

    Handles ``$``, thousands separators, ``%``, parenthesised negatives, and FinQA
    ``const_`` tokens. ``%`` divides by 100 (e.g. ``12.5%`` -> ``0.125``), matching
    both FinQA's reference executor and real-world convention.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith(_CONST_PREFIX):
        body = s[len(_CONST_PREFIX) :]
        if body == "m1":
            return Decimal(-1)
        try:
            return Decimal(body)
        except InvalidOperation:
            return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    percent = "%" in s
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    if not s:
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    if percent:
        value = value / 100
    return -value if negative else value


def answers_match(
    predicted: Decimal | bool | str | float | int | None,
    gold: Decimal | bool | str | float | int | None,
    *,
    rel_tol: Decimal = _REL_TOL,
    abs_tol: Decimal = _ABS_TOL,
) -> bool:
    """True if ``predicted`` matches ``gold`` (yes/no exactly, numbers within tolerance)."""
    pred_bool = _as_bool(predicted)
    gold_bool = _as_bool(gold)
    if pred_bool is not None or gold_bool is not None:
        return pred_bool is not None and pred_bool == gold_bool

    pred_num = _as_decimal(predicted)
    gold_num = _as_decimal(gold)
    if pred_num is None or gold_num is None:
        return False
    diff = abs(pred_num - gold_num)
    return diff <= max(abs_tol, rel_tol * max(abs(pred_num), abs(gold_num)))


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "true"}:
            return True
        if v in {"no", "false"}:
            return False
    return None


def _as_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        return str_to_num(value)
    return None
