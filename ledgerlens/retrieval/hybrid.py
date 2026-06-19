"""Hybrid retrieval: fuse lexical (BM25) and dense (vector) rankings with RRF.

Reciprocal Rank Fusion combines the two ranked lists by rank position, so neither
retriever's score scale matters — robust and parameter-light.
"""

from __future__ import annotations

from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk
from ledgerlens.retrieval.vector import VectorIndex

_RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[Chunk]], *, rrf_k: int = _RRF_K) -> list[Chunk]:
    """Fuse ranked chunk lists into one ranking via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    items: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            items[chunk.chunk_id] = chunk
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [items[cid] for cid in ordered]


class HybridRetriever:
    """Run BM25 and vector retrieval, then fuse with RRF."""

    def __init__(self, bm25: Bm25Index, vector: VectorIndex) -> None:
        self.bm25 = bm25
        self.vector = vector

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        pool = max(k * 3, 10)
        lexical = self.bm25.search(query, k=pool)
        dense = self.vector.search(query, k=pool)
        return reciprocal_rank_fusion([lexical, dense])[:k]
