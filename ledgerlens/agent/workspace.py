"""Per-company filing workspace + a registry that resolves tickers to filings.

A workspace holds what the agent's tools need for one company: a BM25 index over the latest
10-K's chunks, plus its XBRL facts. The registry serves a **committed demo bundle** if one
exists (so the hosted demo needs no live SEC calls — SEC blocks shared cloud IPs), and
otherwise resolves the ticker → CIK and fetches live (local dev / other tickers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from ledgerlens.ingest.edgar import EdgarClient
from ledgerlens.ingest.filing import parse_filing_html
from ledgerlens.ingest.xbrl import XbrlFact, parse_company_facts
from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk, chunk_filing

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_BUNDLE_DIR = Path(__file__).parent / "demo_data"


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


def load_workspace_bundle(path: Path) -> FilingWorkspace:
    """Build a workspace from a committed demo bundle JSON (no network)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(chunk_id=c["id"], text=c["text"], kind=c.get("kind", "text")) for c in data["chunks"]
    ]
    facts = [
        XbrlFact(
            taxonomy=f["taxonomy"],
            concept=f["concept"],
            unit=f["unit"],
            value=Decimal(f["value"]),
            period_end=f["period_end"],
            fiscal_year=f["fiscal_year"],
            fiscal_period=f["fiscal_period"],
            form=f["form"],
        )
        for f in data["facts"]
    ]
    return FilingWorkspace(
        ticker=data["ticker"],
        cik=data["cik"],
        title=data["title"],
        filing_url=data["filing_url"],
        bm25=Bm25Index(chunks),
        facts=facts,
    )


class WorkspaceRegistry:
    """Serve a committed demo bundle if present, else resolve ticker → CIK and fetch live."""

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
        bundle = _BUNDLE_DIR / f"{key}.json"
        if bundle.exists():
            self._cache[key] = load_workspace_bundle(bundle)
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
