"""Tests for EDGAR ingestion: HTML table/section parsing, filing lookup, XBRL facts."""

from __future__ import annotations

from decimal import Decimal

from ledgerlens.ingest.edgar import find_latest_filing
from ledgerlens.ingest.filing import parse_filing_html
from ledgerlens.ingest.xbrl import match_value, parse_company_facts

_HTML = """
<html><body>
<p>Item 7. Management's Discussion and Analysis of Financial Condition.</p>
<p>The following table sets forth net revenue (in millions):</p>
<table>
  <tr><th></th><th>2023</th><th>2022</th></tr>
  <tr><td>Net revenue</td><td>$</td><td>5,829</td><td>5,735</td></tr>
  <tr><td>Cost of sales</td><td></td><td>(1,000)</td><td>(900)</td></tr>
</table>
<p>Item 8. Financial Statements and Supplementary Data.</p>
<p>Additional detail follows here.</p>
</body></html>
"""

_SUBMISSIONS = {
    "cik": 320193,
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0000320193-24-000001",
                "0000320193-23-000106",
                "0000320193-23-000077",
            ],
            "primaryDocument": ["x.htm", "aapl-20230930.htm", "y.htm"],
            "filingDate": ["2024-01-01", "2023-11-03", "2023-08-04"],
        }
    },
}

_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2023-09-30", "val": 383285000000, "fy": 2023, "form": "10-K"},
                        {"end": "2022-09-24", "val": 394328000000, "fy": 2022, "form": "10-K"},
                    ]
                }
            }
        }
    }
}


def test_parse_filing_html_extracts_unit_aware_table() -> None:
    parsed = parse_filing_html(_HTML, cik="320193")
    assert len(parsed.tables) >= 1
    extracted = parsed.tables[0]
    assert extracted.unit_hint == "in millions"
    assert any("Net revenue" in cell for row in extracted.table.rows for cell in row)


def test_parse_filing_html_splits_items() -> None:
    parsed = parse_filing_html(_HTML)
    items = {section.item for section in parsed.sections}
    assert "Item 7" in items
    assert "Item 8" in items


def test_find_latest_filing_picks_most_recent_10k() -> None:
    ref = find_latest_filing(_SUBMISSIONS, form="10-K")
    assert ref is not None
    assert ref.accession == "0000320193-23-000106"
    assert ref.primary_document == "aapl-20230930.htm"
    assert ref.url.endswith("/320193/000032019323000106/aapl-20230930.htm")


def test_xbrl_facts_parse_and_match() -> None:
    facts = parse_company_facts(_FACTS)
    assert len(facts) == 2
    matches = match_value(facts, Decimal("383285000000"))
    assert len(matches) == 1
    assert matches[0].fiscal_year == 2023
    assert matches[0].concept == "Revenues"
