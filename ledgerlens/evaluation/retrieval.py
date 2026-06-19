"""Retrieval recall@k on FinQA via BM25.

Each example's context is built as chunks using FinQA's own ids (``text_i`` for the
i-th pre+post sentence, ``table_i`` for the i-th table row), relevance is taken from
``gold_inds``, and we report recall@k / hit@k. ``gold_coverage`` confirms our chunk
ids line up with FinQA's annotations. On FinQA's single-page contexts retrieval is
comparatively easy; the harder real-filing / FinanceBench eval is the next step.
No LLM is involved.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ledgerlens.retrieval.bm25 import Bm25Index
from ledgerlens.retrieval.chunk import Chunk, serialize_table_row

_K_VALUES = (1, 3, 5)


@dataclass
class RetrievalReport:
    """recall@k / hit@k plus how much gold our corpus actually represents."""

    total: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    gold_coverage: float = 0.0


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
    data: list[dict], *, k_values: tuple[int, ...] = _K_VALUES
) -> RetrievalReport:
    """Run BM25 retrieval over each item and aggregate recall@k / hit@k."""
    max_k = max(k_values)
    recall_sum = dict.fromkeys(k_values, 0.0)
    hit_sum = dict.fromkeys(k_values, 0.0)
    gold_seen = 0
    gold_present = 0
    report = RetrievalReport()
    for item in data:
        qa = item.get("qa") or {}
        question = qa.get("question", "")
        gold_all = set((qa.get("gold_inds") or {}).keys())
        if not gold_all or not question:
            continue
        chunks = build_finqa_chunks(item)
        ids = {c.chunk_id for c in chunks}
        gold = gold_all & ids
        gold_seen += len(gold_all)
        gold_present += len(gold)
        if not gold:
            continue
        report.total += 1
        ranked_ids = [c.chunk_id for c in Bm25Index(chunks).search(question, k=max_k)]
        for k in k_values:
            topk = set(ranked_ids[:k])
            recall_sum[k] += len(gold & topk) / len(gold)
            hit_sum[k] += 1.0 if gold & topk else 0.0
    n = report.total or 1
    report.recall = {k: recall_sum[k] / n for k in k_values}
    report.hit = {k: hit_sum[k] / n for k in k_values}
    report.gold_coverage = gold_present / gold_seen if gold_seen else 0.0
    return report


def run_finqa_retrieval_eval(
    path: str | Path, *, k_values: tuple[int, ...] = _K_VALUES, limit: int | None = None
) -> RetrievalReport:
    """Load a FinQA JSON file and evaluate BM25 retrieval over it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    return evaluate_finqa_items(data, k_values=k_values)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m ledgerlens.evaluation.retrieval <finqa.json> [limit]")
        return 2
    path = Path(argv[0])
    limit = int(argv[1]) if len(argv) > 1 else None
    report = run_finqa_retrieval_eval(path, limit=limit)
    cov = report.gold_coverage
    print(f"BM25 retrieval on FinQA: {path}  (n={report.total}, gold_coverage={cov:.1%})")
    for k in sorted(report.recall):
        print(f"  recall@{k}: {report.recall[k]:.1%}   hit@{k}: {report.hit[k]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
