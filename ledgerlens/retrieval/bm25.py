"""Lexical retrieval over chunks with BM25.

Lexical matching is load-bearing for financial retrieval: line-item names and years
(e.g. "2023", "net revenue") are exact-match signals that dense embeddings blur.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ledgerlens.retrieval.chunk import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9.]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer (keeps decimals like 12.5)."""
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """A BM25 index over a fixed list of chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the top-``k`` chunks ranked by BM25 score."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        paired = zip(self.chunks, scores, strict=True)
        ranked = sorted(paired, key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in ranked[:k]]
