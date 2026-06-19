"""Parse a real 10-K HTML filing into sections and unit-aware tables.

EDGAR's canonical 10-K is HTML/iXBRL, so financial tables are real ``<table>``
elements parsed deterministically with BeautifulSoup — no OCR, no column-boundary
guessing. Each table carries its unit hint ("in millions") and a nearby caption, so
a raw number keeps the scale context that makes it meaningful.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import Tag

from ledgerlens.executor.dsl import Table

_UNIT_RE = re.compile(r"in\s+(thousands|millions|billions)", re.IGNORECASE)
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}[A-Z]?)\.", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass
class ExtractedTable:
    """A table lifted from the filing, with its scale/caption context."""

    caption: str
    unit_hint: str
    table: Table


@dataclass
class Section:
    """A 10-K item section (e.g. Item 7 — MD&A)."""

    item: str
    text: str


@dataclass
class ParsedFiling:
    """A parsed filing: full text, item sections, and unit-aware tables."""

    cik: str
    text: str
    sections: list[Section] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)


def parse_filing_html(html: str, *, cik: str = "") -> ParsedFiling:
    """Parse 10-K HTML into a :class:`ParsedFiling`."""
    with warnings.catch_warnings():
        # EDGAR filings are iXBRL; we intentionally parse them as HTML for the tables.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for junk in soup(["script", "style"]):
        junk.decompose()

    tables: list[ExtractedTable] = []
    for tag in soup.find_all("table"):
        table = _extract_table(tag)
        if table is None or len(table.rows) < 2:
            continue
        tables.append(
            ExtractedTable(caption=_caption(tag), unit_hint=_unit_hint(tag), table=table)
        )

    text = _clean(soup.get_text(" "))
    return ParsedFiling(cik=cik, text=text, sections=_split_items(text), tables=tables)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip()


def _extract_table(tag: Tag) -> Table | None:
    rows: list[list[str]] = []
    for tr in tag.find_all("tr"):
        cells = [_clean(cell.get_text(" ")) for cell in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]  # drop empty layout/spacer cells
        if cells:
            rows.append(cells)
    return Table(rows=rows) if rows else None


def _unit_hint(tag: Tag) -> str:
    previous = tag.find_previous(string=_UNIT_RE)
    match = _UNIT_RE.search(str(previous)) if previous else _UNIT_RE.search(tag.get_text(" "))
    return f"in {match.group(1).lower()}" if match else ""


def _caption(tag: Tag) -> str:
    previous = tag.find_previous(["b", "strong", "h1", "h2", "h3", "h4", "p"])
    return _clean(previous.get_text(" "))[:120] if previous else ""


def _split_items(text: str) -> list[Section]:
    matches = list(_ITEM_RE.finditer(text))
    best: dict[str, Section] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = f"Item {match.group(1).upper()}"
        body = text[match.start() : end].strip()
        # Keep the longest span per item — the real section, not the table-of-contents row.
        if item not in best or len(body) > len(best[item].text):
            best[item] = Section(item=item, text=body)
    return list(best.values())
