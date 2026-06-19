"""Dense vector retrieval over chunks.

Exact cosine similarity with NumPy — correct and dependency-light at the corpus sizes
here (a filing's chunks, a FinQA context). The interface is identical to a FAISS-backed
index, so FAISS is a drop-in for larger corpora (documented as the scale-up path).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ledgerlens.retrieval.chunk import Chunk

EmbedFn = Callable[[list[str]], list[list[float]]]


class VectorIndex:
    """A dense index that embeds chunk texts once and ranks by cosine similarity."""

    def __init__(self, chunks: list[Chunk], embed_fn: EmbedFn) -> None:
        self.chunks = list(chunks)
        self._embed = embed_fn
        if self.chunks:
            matrix = np.asarray(self._embed([c.text for c in self.chunks]), dtype=float)
            self._matrix = _normalize(matrix)
        else:
            self._matrix = np.zeros((0, 0))

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the top-``k`` chunks ranked by cosine similarity to ``query``."""
        if not self.chunks:
            return []
        query_vec = _normalize(np.asarray(self._embed([query]), dtype=float))[0]
        sims = self._matrix @ query_vec
        order = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in order]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)
