"""Table-aware hybrid retrieval (P3).

FAISS + BM25 as the primary engine (lexical is load-bearing for numbers/periods),
with Azure AI Search hybrid as the deployable showcase, behind one interface.
"""

from ledgerlens.retrieval.bm25 import Bm25Index, tokenize
from ledgerlens.retrieval.chunk import Chunk, chunk_filing, serialize_table_row
from ledgerlens.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from ledgerlens.retrieval.vector import VectorIndex

__all__ = [
    "Bm25Index",
    "Chunk",
    "HybridRetriever",
    "VectorIndex",
    "chunk_filing",
    "reciprocal_rank_fusion",
    "serialize_table_row",
    "tokenize",
]
