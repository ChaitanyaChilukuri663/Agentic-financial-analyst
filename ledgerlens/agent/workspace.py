"""Per-company filing workspace + a registry that resolves tickers to filings.

A workspace holds everything the agent's tools need for one company: a BM25 index over the
latest 10-K's chunks, and the company's XBRL facts. The registry resolves a ticker to a CIK
(via SEC's company_tickers.json) and builds/caches a workspace on demand — which is what
lets the agent work across multiple companies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ledgerlens.ingest.edgar import EdgarClient
from ledgerlens.ingest.filing import parse_filing_html
from ledgerlens.ingest.xbrl import XbrlFact, parse_company_facts
from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import chunk_filing

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class FilingWorkspace:
    """The evidence for one company's latest 10-K."""

    ticker: str
    cik: str
    title: str
    filing_url: str
    bm25: Bm25Index
    facts: list[XbrlFact]

    def retrieve(self, query: str, k: int = 8) -> str:
        return "\n".join(chunk.text for chunk in self.bm25.search(query, k))


class WorkspaceRegistry:
    """Resolves tickers -> CIK and builds/caches a :class:`FilingWorkspace` per company."""

    def __init__(self, edgar: EdgarClient) -> None:
        self.edgar = edgar
        self._tickers: dict[str, dict] | None = None
        self._cache: dict[str, FilingWorkspace] = {}

    def _ticker_map(self) -> dict[str, dict]:
        if self._tickers is None:
            raw = json.loads(self.edgar.fetch_text(_TICKERS_URL))
            self._tickers = {row["ticker"].upper(): row for row in raw.values()}
        return self._tickers

    def get(self, ticker: str) -> FilingWorkspace | None:
        key = (ticker or "").strip().upper()
        if key in self._cache:
            return self._cache[key]
        info = self._ticker_map().get(key)
        if info is None:
            return None
        cik = str(info["cik_str"])
        ref = self.edgar.latest_10k(cik)
        if ref is None:
            return None
        parsed = parse_filing_html(self.edgar.fetch_text(ref.url), cik=cik)
        workspace = FilingWorkspace(
            ticker=key,
            cik=cik,
            title=info.get("title", ""),
            filing_url=ref.url,
            bm25=Bm25Index(chunk_filing(parsed, filing_id=key)),
            facts=parse_company_facts(self.edgar.company_facts(cik)),
        )
        self._cache[key] = workspace
        return workspace
