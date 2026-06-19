"""Minimal, polite EDGAR client: fetch filings + XBRL facts with on-disk caching.

SEC requires a descriptive ``User-Agent`` and rate-limits to ~10 requests/second; this
client sets the UA, throttles, and caches every response to disk so any document is
fetched at most once.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ledgerlens.net import build_http_client

_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_MIN_INTERVAL_S = 0.15


@dataclass
class FilingRef:
    """A pointer to one filing and its primary document."""

    cik: str
    accession: str
    form: str
    filing_date: str
    primary_document: str
    url: str


def find_latest_filing(submissions: dict, form: str = "10-K") -> FilingRef | None:
    """Locate the most recent filing of ``form`` in a submissions JSON payload."""
    cik = str(submissions.get("cik", "")).lstrip("0") or "0"
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, value in enumerate(forms):
        if value == form:
            accession = recent["accessionNumber"][i]
            document = recent["primaryDocument"][i]
            return FilingRef(
                cik=cik,
                accession=accession,
                form=value,
                filing_date=recent["filingDate"][i],
                primary_document=document,
                url=_ARCHIVE.format(cik=cik, acc=accession.replace("-", ""), doc=document),
            )
    return None


class EdgarClient:
    """Caching, rate-limited HTTP client for SEC EDGAR."""

    def __init__(
        self,
        user_agent: str,
        *,
        cache_dir: str | Path = "data/edgar_cache",
        use_os_truststore: bool = True,
        timeout_s: float = 30.0,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self._client = build_http_client(
            use_os_truststore=use_os_truststore,
            timeout_s=timeout_s,
            headers={"User-Agent": user_agent},
        )
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval_s
        self._last_request = 0.0

    def _get(self, url: str) -> bytes:
        cache_key = self._cache / f"{hashlib.sha256(url.encode()).hexdigest()}.cache"
        if cache_key.exists():
            return cache_key.read_bytes()
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        response = self._client.get(url, follow_redirects=True)
        self._last_request = time.monotonic()
        response.raise_for_status()
        cache_key.write_bytes(response.content)
        return response.content

    def submissions(self, cik: str | int) -> dict:
        """Return the submissions JSON (recent filings index) for a CIK."""
        return json.loads(self._get(_SUBMISSIONS.format(cik=int(cik))))

    def company_facts(self, cik: str | int) -> dict:
        """Return the XBRL companyfacts JSON (every reported numeric fact) for a CIK."""
        return json.loads(self._get(_COMPANY_FACTS.format(cik=int(cik))))

    def fetch_text(self, url: str) -> str:
        """Fetch a document (e.g. a filing's primary HTML) as text."""
        return self._get(url).decode("utf-8", errors="replace")

    def latest_10k(self, cik: str | int) -> FilingRef | None:
        """Return a reference to the company's most recent 10-K, if any."""
        return find_latest_filing(self.submissions(cik), form="10-K")
