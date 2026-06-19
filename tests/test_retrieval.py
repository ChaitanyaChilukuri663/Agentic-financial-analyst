"""Tests for chunking, BM25 / vector / hybrid retrieval, and the recall@k harness."""

from __future__ import annotations

from ledgerlens.evaluation.retrieval import evaluate_finqa_items
from ledgerlens.executor.dsl import Table
from ledgerlens.ingest.filing import ExtractedTable, ParsedFiling, Section
from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk, chunk_filing, serialize_table_row
from ledgerlens.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from ledgerlens.retrieval.vector import VectorIndex

_VOCAB = ["revenue", "cost", "europe", "2023"]


def _chunks() -> list[Chunk]:
    return [
        Chunk("a", "net revenue increased in 2023 to 5829 million", "text"),
        Chunk("b", "the company hired new staff in europe", "text"),
        Chunk("c", "cost of sales was 1000 in 2023", "text"),
    ]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(term in text.lower()) for term in _VOCAB] for text in texts]


def test_serialize_table_row_is_self_describing() -> None:
    text = serialize_table_row(["company", "2023", "2022"], ["amex", "637", "647"])
    assert "amex" in text
    assert "2023" in text
    assert "637" in text


def test_bm25_ranks_relevant_chunk_first() -> None:
    top = Bm25Index(_chunks()).search("what was net revenue in 2023", k=1)
    assert top[0].chunk_id == "a"


def test_vector_index_with_fake_embedder() -> None:
    top = VectorIndex(_chunks(), _fake_embed).search("cost", k=1)
    assert top[0].chunk_id == "c"


def test_rrf_fusion_rewards_agreement() -> None:
    chunks = _chunks()
    fused = reciprocal_rank_fusion([[chunks[0], chunks[1]], [chunks[2], chunks[0]]])
    assert fused[0].chunk_id == "a"  # appears in both lists -> highest fused score


def test_hybrid_retriever_runs() -> None:
    retriever = HybridRetriever(Bm25Index(_chunks()), VectorIndex(_chunks(), _fake_embed))
    top = retriever.search("net revenue 2023", k=2)
    assert any(c.chunk_id == "a" for c in top)


def test_chunk_filing_produces_table_and_text_chunks() -> None:
    parsed = ParsedFiling(
        cik="1",
        text="MD&A long text here",
        sections=[Section(item="Item 7", text="Management discussion and analysis text.")],
        tables=[
            ExtractedTable(
                caption="Net revenue",
                unit_hint="in millions",
                table=Table(rows=[["item", "2023", "2022"], ["net revenue", "5829", "5735"]]),
            )
        ],
    )
    chunks = chunk_filing(parsed, filing_id="f1")
    kinds = {c.kind for c in chunks}
    assert {"table_row", "text"} <= kinds
    row_chunk = next(c for c in chunks if c.kind == "table_row")
    assert "net revenue" in row_chunk.text
    assert "in millions" in row_chunk.text


def test_finqa_recall_harness_synthetic() -> None:
    item = {
        "pre_text": ["the company grew", "net revenue was 5829 in 2023"],
        "post_text": [],
        "table": [["company", "2023"], ["amex", "637"]],
        "qa": {
            "question": "what was net revenue in 2023",
            "gold_inds": {"text_1": "net revenue was 5829 in 2023"},
        },
    }
    report = evaluate_finqa_items([item], k_values=(1, 3))
    assert report.total == 1
    assert report.gold_coverage == 1.0
    assert report.recall[3] == 1.0
