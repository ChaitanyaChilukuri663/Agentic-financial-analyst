# LedgerLens

**A grounded financial-QA engine — the LLM proposes a reasoning *program*; a deterministic engine does the math.**

Analysts dig numbers out of 100–300 page SEC filings and compute ratios by hand. The "just paste it into an LLM" shortcut fails three ways: the filing doesn't fit in the context window, the model **hallucinates figures**, and it **botches arithmetic**. LedgerLens fixes all three:

1. **Retrieve** the exact evidence from real filings (table-aware hybrid RAG).
2. The LLM **proposes a symbolic reasoning program** — operands (each with a citation) + operations — **not the final answer**.
3. A deterministic, pure-Python **executor runs the program** (the LLM never does arithmetic).
4. **Validate** the result on *trustworthy* signals — retrieval score, program validity, numeric sanity, and operand-grounding — then **answer** (with citations and shown steps) **or abstain**.

A Phase 2 **agent** then orchestrates LedgerLens across multiple filings — *an agent that can't fabricate its numbers, because every figure routes through the verified tool.*

> **Status:** early-stage, built in public, phase by phase. This is a portfolio project — honest framing, no overclaiming. The evaluation is the centerpiece; hard numbers land as the phases complete (see **Roadmap**).

## Architecture

```
OFFLINE INGEST: EDGAR 10-Ks (HTML/iXBRL) + FinQA contexts
   → table-aware parse → chunk → embed → vector index (+ XBRL companyfacts as a fact anchor)

ONLINE (per question):
   1) RETRIEVE   hybrid (vector + BM25), top-k  → text chunks + structured table rows  [evidence]
   2) EXTRACT+PLAN  one structured LLM call → { operands (+citations), reasoning program }  (Pydantic-validated)
   3) EXECUTE    deterministic interpreter runs the ops → number
   4) VALIDATE   retrieval score + program validity + numeric sanity + operand-grounding → answer or abstain
   → ANSWER + citations + shown steps

PHASE 2 AGENT: plan → act → observe → revise → terminate, calling LedgerLens as a verified tool over multiple filings
```

## Design decisions (the *why*)

- **The deterministic executor is the crown jewel.** The LLM emits a symbolic program; pure Python executes it. This is what kills arithmetic hallucination.
- **RAG is load-bearing**, not decorative — real 10-Ks exceed the context window and suffer "lost in the middle." We prove it with a RAG-vs-naive-full-context baseline.
- **Real, ugly 10-K ingestion is first-class scope** (HTML/iXBRL parsing + XBRL facts), not a demo afterthought. The differentiator is "survives a real filing," not "works on a clean excerpt."
- **Abstention gates on trustworthy signals** (retrieval/program-validity/numeric-sanity/grounding), **never** LLM self-reported confidence.
- **Hand-rolled core, no LangChain.** Depth in the retriever/executor beats a framework keyword.

## Roadmap

| Phase | Scope | Status |
|------|-------|--------|
| **P0** | Scaffold: package layout, config, multi-provider `llm_client`, ruff + pytest + CI | ✅ done |
| **P1** | Deterministic program executor + FinQA gold-program replay — **reproduces 99.5% of 8,281 FinQA gold programs, zero LLM** ([results](evals/results.md)) | ✅ done |
| **P2** | FinQA extract→plan call wired to the executor → determinism baseline | ⏳ next |
| **P2.5** | Real-10-K ingestion spike (fetch one filing, validate table extraction) | ☐ |
| **P3** | EDGAR HTML/iXBRL ingestion + table-aware hybrid RAG + retrieval eval + FinanceBench stress test | ☐ |
| **P4** | Validation/abstention gates + citations + error taxonomy + baselines → `evals/results.md` | ☐ |
| **P5** | Streamlit + FastAPI demo + free-host deploy + `DEPLOY.md` (Azure scale-up path) | ☐ |
| **P6** | Phase 2 agent loop wrapping LedgerLens as a verified tool + agent telemetry eval | ☐ |

## Tech stack

Python 3.12 · hand-rolled multi-provider `llm_client` (forced-tool-call → Pydantic v2 structured output, embeddings, OS-trust-store SSL) · **Azure OpenAI** `gpt-4o-mini` + `text-embedding-3-small` (GitHub Models / Groq as fallbacks) · FAISS + BM25 hybrid retrieval (Azure AI Search free-tier as the deployable showcase) · pure-Python program-DSL executor · FastAPI + Streamlit · ruff + pytest, CI on GitHub Actions.

## Local setup (Python 3.12)

```bash
git clone <repo-url>
cd "Multi agent finqa"

python3.12 -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env          # then fill in your Azure OpenAI endpoint + key

ruff check .                  # lint (E/F/I/B/UP/ASYNC, line length 100)
pytest                        # unit tests (LLM is mocked; no network or keys needed)
```

`pytest` runs fully offline — the LLM client is mocked. Live provider tests are opt-in (`pytest -m live`) and require real credentials.

## Datasets & attribution

- **FinQA** — Chen et al., 2021 (numerical reasoning over financial reports). Used under its dataset license for evaluation.
- **FinanceBench** — Islam et al., 2023 (open-ended QA over real 10-Ks). Used as a real-filing stress test.

All dataset licenses belong to their respective owners; this repo contains no redistributed dataset content.

## License

MIT (project code). See dataset attribution above for data licensing.
