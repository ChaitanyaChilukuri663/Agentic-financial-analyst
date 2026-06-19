"""Retrieval recall@k on FinQA: BM25 vs dense vs hybrid.

Each example's context is built as chunks using FinQA's own ids (``text_i`` for the
i-th pre+post sentence, ``table_i`` for the i-th table row); relevance comes from
``gold_inds`` and ``gold_coverage`` confirms our ids line up with FinQA. BM25 needs no
LLM; passing an ``embed_fn`` adds dense (cosine) and hybrid (RRF) retrieval. On FinQA's
single-page contexts retrieval is comparatively easy — the harder real-filing /
FinanceBench eval is the next step.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk, serialize_table_row
from ledgerlens.retrieval.hybrid import reciprocal_rank_fusion
from ledgerlens.retrieval.vector import VectorIndex

_K_VALUES = (1, 3, 5)
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class MethodScores:
    """recall@k and hit@k for one retrieval method."""

    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)


@dataclass
class RetrievalReport:
    """Per-method scores plus how much gold our corpus represents."""

    total: int = 0
    gold_coverage: float = 0.0
    methods: dict[str, MethodScores] = field(default_factory=dict)


def build_finqa_chunks(item: dict) -> list[Chunk]:
    """Build context chunks with FinQA's ``text_i`` / ``table_i`` ids."""
    chunks: list[Chunk] = []
    all_text = list(item.get("pre_text", [])) + list(item.get("post_text", []))
    for i, sentence in enumerate(all_text):
        chunks.append(Chunk(chunk_id=f"text_{i}", text=str(sentence), kind="text"))
    table = item.get("table", [])
    header = [str(c) for c in table[0]] if table else []
    for i, row in enumerate(table):
        text = serialize_table_row(header, [str(c) for c in row])
        chunks.append(Chunk(chunk_id=f"table_{i}", text=text, kind="table_row"))
    return chunks


def evaluate_finqa_items(
    data: list[dict], *, k_values: tuple[int, ...] = _K_VALUES, embed_fn: EmbedFn | None = None
) -> RetrievalReport:
    """Run BM25 (and, if ``embed_fn`` is given, dense + hybrid) retrieval and aggregate."""
    methods = ["bm25", "vector", "hybrid"] if embed_fn else ["bm25"]
    recall_sum = {m: dict.fromkeys(k_values, 0.0) for m in methods}
    hit_sum = {m: dict.fromkeys(k_values, 0.0) for m in methods}
    total = 0
    gold_seen = 0
    gold_present = 0
    for item in data:
        qa = item.get("qa") or {}
        question = qa.get("question", "")
        gold_all = set((qa.get("gold_inds") or {}).keys())
        if not gold_all or not question:
            continue
        chunks = build_finqa_chunks(item)
        gold = gold_all & {c.chunk_id for c in chunks}
        gold_seen += len(gold_all)
        gold_present += len(gold)
        if not gold:
            continue
        total += 1
        ranked = _rank_all(chunks, question, methods, embed_fn)
        for method in methods:
            for k in k_values:
                topk = set(ranked[method][:k])
                recall_sum[method][k] += len(gold & topk) / len(gold)
                hit_sum[method][k] += 1.0 if gold & topk else 0.0
    n = total or 1
    coverage = gold_present / gold_seen if gold_seen else 0.0
    report = RetrievalReport(total=total, gold_coverage=coverage)
    report.methods = {
        method: MethodScores(
            recall={k: recall_sum[method][k] / n for k in k_values},
            hit={k: hit_sum[method][k] / n for k in k_values},
        )
        for method in methods
    }
    return report


def _rank_all(
    chunks: list[Chunk], question: str, methods: list[str], embed_fn: EmbedFn | None
) -> dict[str, list[str]]:
    bm25_full = Bm25Index(chunks).search(question, k=len(chunks))
    ranked = {"bm25": [c.chunk_id for c in bm25_full]}
    if embed_fn is not None:
        vector_full = VectorIndex(chunks, embed_fn).search(question, k=len(chunks))
        fused = reciprocal_rank_fusion([bm25_full, vector_full])
        ranked["vector"] = [c.chunk_id for c in vector_full]
        ranked["hybrid"] = [c.chunk_id for c in fused]
    return ranked


def run_finqa_retrieval_eval(
    path: str | Path,
    *,
    k_values: tuple[int, ...] = _K_VALUES,
    limit: int | None = None,
    embed_fn: EmbedFn | None = None,
) -> RetrievalReport:
    """Load a FinQA JSON file and evaluate retrieval over it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    return evaluate_finqa_items(data, k_values=k_values, embed_fn=embed_fn)


def _throttled(fn: EmbedFn, min_interval: float = 1.2) -> EmbedFn:
    """Space out embed calls to stay under low-tier rate limits (retries are the backstop)."""
    state = {"last": 0.0}

    def wrapped(texts: list[str]) -> list[list[float]]:
        wait = min_interval - (time.monotonic() - state["last"])
        if wait > 0:
            time.sleep(wait)
        try:
            return fn(texts)
        finally:
            state["last"] = time.monotonic()

    return wrapped


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.retrieval <finqa.json> [limit] [dense]")
        return 2
    path = Path(argv[0])
    limit = next((int(a) for a in argv[1:] if a.isdigit()), None)
    embed_fn: EmbedFn | None = None
    if "dense" in argv:
        from ledgerlens.llm import LLMClient

        embed_fn = _throttled(LLMClient().embed)
    report = run_finqa_retrieval_eval(path, limit=limit, embed_fn=embed_fn)
    cov = report.gold_coverage
    print(f"Retrieval on FinQA: {path}  (n={report.total}, gold_coverage={cov:.1%})")
    for method, scores in report.methods.items():
        cells = "  ".join(
            f"r@{k} {scores.recall[k]:.1%}/h@{k} {scores.hit[k]:.1%}" for k in sorted(scores.recall)
        )
        print(f"  {method:<7}: {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
