"""EDGAR ingestion (P3).

HTML/iXBRL-first parsing of real 10-K filings (unit-aware tables, footnote scale
context) plus XBRL companyfacts as a ground-truth anchor for operand-grounding.
"""

from ledgerlens.ingest.edgar import EdgarClient, FilingRef, find_latest_filing
from ledgerlens.ingest.filing import ExtractedTable, ParsedFiling, Section, parse_filing_html
from ledgerlens.ingest.xbrl import XbrlFact, match_value, parse_company_facts

__all__ = [
    "EdgarClient",
    "ExtractedTable",
    "FilingRef",
    "ParsedFiling",
    "Section",
    "XbrlFact",
    "find_latest_filing",
    "match_value",
    "parse_company_facts",
    "parse_filing_html",
]
