"""Tests for FinQA-compatible number parsing and answer comparison."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ledgerlens.executor.numbers import answers_match, str_to_num


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234", Decimal("1234")),
        ("$1,234.5", Decimal("1234.5")),
        ("(123)", Decimal("-123")),
        ("($1,000)", Decimal("-1000")),
        ("50%", Decimal("0.5")),
        ("12.5%", Decimal("0.125")),
        ("const_1000000", Decimal("1000000")),
        ("const_100", Decimal("100")),
        ("const_m1", Decimal("-1")),
        ("  -42 ", Decimal("-42")),
        ("n/a", None),
        ("", None),
        (None, None),
    ],
)
def test_str_to_num(text: str | None, expected: Decimal | None) -> None:
    assert str_to_num(text) == expected


def test_answers_match_within_tolerance() -> None:
    assert answers_match(Decimal("0.1413"), 0.1413)
    assert answers_match(Decimal("100.00001"), 100.0)
    assert not answers_match(Decimal("100"), 110)


def test_answers_match_yes_no() -> None:
    assert answers_match(True, "yes")
    assert answers_match(False, "no")
    assert not answers_match(True, "no")
    assert not answers_match(True, 1)


def test_answers_match_handles_strings() -> None:
    assert answers_match("14.1%", Decimal("0.141"))
    assert not answers_match("n/a", Decimal("1"))
