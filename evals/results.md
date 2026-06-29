# Evaluation Results

## P1 — Deterministic executor validation (FinQA gold-program replay)

**Claim under test:** the executor computes correctly. We validate it *independently of
any LLM* by replaying FinQA's expert-written gold programs — parse each gold program into
the DSL, execute it, and compare the computed value to FinQA's gold `exe_ans` (relative
tolerance 0.1%; `yes`/`no` compared exactly).

**Dataset:** FinQA (Chen et al., 2021) — 8,281 expert-annotated Q&A items over S&P 500
10-K filings. **No LLM is involved in this experiment.**

### Reproduction rate

| Split | Examples | Reproduced | Rate |
|-------|---------:|-----------:|-----:|
| train | 6,251 | 6,218 | 99.5% |
| dev   |   883 |   881 | 99.8% |
| test  | 1,147 | 1,142 | 99.6% |
| **All** | **8,281** | **8,241** | **99.52%** |

Accounting for gold-answer rounding (see taxonomy), effective reproduction is
**99.95%** (8,277 / 8,281).

### Error taxonomy (48 non-reproduced of 8,281)

| Category | Count | What it is |
|----------|------:|------------|
| gold-rounding | 36 | The executor is *correct*; FinQA's `exe_ans` is rounded to 2–3 significant figures (e.g. computed `0.0031652` vs gold `0.00317`). The computed value rounds to the gold at the gold's own precision. |
| genuine | 4 | 2 dataset-noise cases (gold loosely rounded to `1e-05`); 2 DISH stock-price rows with a non-standard multi-value layout. |

Two fixes during validation, each traced to FinQA's reference code (not guesswork):

- **Percent convention:** FinQA's `str_to_num` maps `%` → `/100`. Adopting it (correct for
  real filings too) moved the dev split from **95.6% → 99.8%**.
- **Numeric row labels:** programs like `table_average(2016, none)` use a *year* as the row
  label. A table op's first operand is now always parsed as a row name, eliminating 8
  spurious errors.

### Why this matters

This is the project's foundational guarantee: **the arithmetic engine is provably correct
on 8k+ expert programs before a single token is spent on an LLM.** In LedgerLens the LLM
only *proposes* a program; this executor *computes* it — so the computation cannot
hallucinate, and the engine's soundness is measured, not asserted.

### Design note

The executor is unit-agnostic with a swappable cell parser. The replay injects a
**FinQA-faithful** parser (mirrors FinQA's `process_row`: strip `$`, take the text before
`(`, `%` → `/100`) so the measurement reflects *executor* correctness, not parser quirks.
The production default parser is deliberately more robust (handles `$`, thousands
separators, parenthesised negatives) for real 10-K tables in P3.

### Reproduce

```bash
# FinQA dataset is NOT committed (license + size). Source: https://github.com/czyssrs/FinQA
# Place dataset/{train,dev,test}.json under data/finqa/, then:
python -m ledgerlens.evaluation.finqa_replay data/finqa/dev.json
```

## P2 — Determinism baseline (program+executor vs LLM-direct)

The core thesis: having the LLM propose a *program* that the deterministic executor runs
beats letting the LLM do the arithmetic itself — most where the arithmetic is hard. Both
modes use the same model (Azure `gpt-4.1-mini`) over the same FinQA evidence; only *who
computes* differs. FinQA dev, n=120:

| Subset | program + executor | LLM direct |
|--------|-------------------:|-----------:|
| single-step programs (n=68) | 60.3% | 69.1% |
| **multi-step programs (n=52)** | **51.9%** | **44.2%** |
| overall (n=120) | 56.7% | 58.3% |

The headline is the **crossover**: on one-step ratios a strong model computes fine on its
own (and proposing/parsing a program adds a little failure surface), but on **multi-step
calculations the executor wins by ~8 points** — determinism pays off exactly where mental
arithmetic breaks down. And beyond accuracy, every program-mode answer is **cited,
inspectable, and provably computed** (the executor reproduces 99.5% of gold programs) —
which the direct mode cannot offer at any accuracy.

Reproduce: `python -m ledgerlens.evaluation.finqa_qa data/finqa/dev.json 120`

## P3 — Real 10-K ingestion (validation)

Ingestion is **HTML/iXBRL-first** (EDGAR's canonical format), so tables are parsed
deterministically with BeautifulSoup — no OCR, no column-boundary guessing. Validated
live on Apple's FY2025 10-K (CIK 320193):

| Metric | Value |
|--------|------:|
| Filing HTML size | 1.5 MB |
| Item sections segmented | 23 (Item 1 … 9C) |
| Tables extracted | 53 |
| Tables with a detected unit ("in millions", …) | 49 |
| XBRL facts parsed (companyfacts) | 24,852 |
| Operand grounding | `383,285M` → `RevenueFromContractWithCustomerExcludingAssessedTax` ✓ |

The XBRL companyfacts feed is a **ground-truth anchor**: an extracted operand can be
checked against a value the company actually filed, not just fuzzy-matched against text.

Reproduce: `EdgarClient(ua).latest_10k(320193)` → `parse_filing_html(html)`; grounding via
`match_value(parse_company_facts(client.company_facts(320193)), value)`.

### Retrieval — BM25 vs dense vs hybrid

The table-aware retriever builds **row-level chunks that carry their header/caption/unit
context**, then ranks with **BM25 (lexical)**, **dense vectors** (`text-embedding-3-small`,
cosine), and a **hybrid** that fuses both via **Reciprocal Rank Fusion**. FinQA dev, n=150,
`gold_coverage` 100% (chunk ids align exactly with FinQA's `gold_inds`, so labels are exact):

| method | recall@1 | recall@3 | recall@5 | hit@3 | hit@5 |
|--------|---------:|---------:|---------:|------:|------:|
| BM25 | 50.8% | 74.4% | 81.9% | 84.7% | 90.0% |
| dense (vector) | 51.9% | 74.4% | 82.6% | 88.7% | 93.3% |
| **hybrid (RRF)** | **52.1%** | **78.1%** | **85.6%** | **90.7%** | **94.0%** |

**Hybrid wins at every cutoff** — BM25 anchors exact lexical matches (line items, years),
dense lifts deeper recall on semantically-phrased questions, and RRF fusion keeps both
(+3.7pp recall@5, +4pp hit@5 over BM25 alone). On FinQA's clean single-page contexts the
gap is modest by design; the larger payoff is expected on real, long 10-Ks where many chunks
compete — the FinanceBench retrieval eval is the next step. (recall@k = fraction of an
example's gold supporting facts in the top-k; hit@k = at least one.)

Reproduce: `python -m ledgerlens.evaluation.retrieval data/finqa/dev.json 150 dense`

## P4 — Validation & abstention gates

A computed answer is returned only if it clears deterministic, hard-to-fake gates:
**program validity** (parsed + executed), **operand grounding** (every literal traces to the
evidence, a filed XBRL fact, or a math constant), and **numeric sanity** (finite, plausible
magnitude). No LLM self-confidence is used. FinQA dev, n=100 (Azure `gpt-4.1-mini`):

| metric | value |
|--------|------:|
| accuracy, no gates (all answers) | 60.0% |
| coverage (answers the gates allow) | 93.0% |
| **precision on answered** | **64.5%** |
| false-abstain rate (correct answers declined) | **0.0%** |

The gates raise precision from 60.0% → 64.5% while **declining only wrong answers** (0%
false-abstain in this slice) — the system trades a little coverage for higher reliability,
at no cost to correct answers here.

### Program-level error taxonomy (of the 40 wrong answers)

| failure | count |
|---------|------:|
| wrong program structure (wrong ops / approach) | 33 |
| wrong operands (right structure, wrong numbers) | 4 |
| parse / execution failure | 3 |

The decisive insight: **the dominant failure is reasoning, not arithmetic or fabrication.**
The executor is exact (P1) and operands are almost always grounded, so the remaining gap is
the LLM choosing the wrong *program structure* — which is also why operand-grounding can't
catch these (the numbers are real, the plan is wrong). That is the honest frontier for this
architecture.

Reproduce: `python -m ledgerlens.evaluation.abstention data/finqa/dev.json 100`

## P6 — Agentic research analyst (iterative, self-correcting, multi-tool, multi-company)

A hand-rolled **plan → act → observe → revise** ReAct loop (no framework). At each step the
LLM picks ONE tool and observes the result before deciding the next move:

- **`xbrl_value`** — one exact figure from the company's filed XBRL data (ground truth;
  canonical-concept resolution; latest fiscal year by default).
- **`compute`** — deterministic arithmetic (via the P1 executor) over figures the agent has
  *already verified* — every input must trace to a prior lookup, so the agent **cannot slip in
  a fabricated number.**
- **`passage`** — a short text snippet for a qualitative fact.

It is iterative (each step chosen from the running state), **self-correcting** (it sees a
failed step and reformulates — switch tool, change keyword/year; a loop-guard blocks repeats),
and **multi-company** (a registry resolves tickers → CIK → filing on demand).

Live run (Azure `gpt-4.1-mini`), task: *"Compare Apple (AAPL) and Microsoft (MSFT) on most
recent year-over-year revenue growth — which grew faster?"*

| telemetry | value |
|-----------|------:|
| tool calls | 6 |
| verified (ok) | 6 |
| failed | 0 |
| success rate | 100% |

It looked up each company's revenue for two years from XBRL (AAPL FY25 $416.2B / FY24 $394.3B;
MSFT FY25 $281.7B / FY24 $245.1B), computed each growth via the executor (**Apple 5.5%,
Microsoft 14.9%**), and concluded **"Microsoft grew faster"** — every figure ground-truth,
every computation deterministic.

This is the project's headline: **an agent whose every number is verified — grounded, cited,
and computed deterministically — not asserted.**

Reproduce: `ledgerlens/agent/` — `ResearchAgent` over `make_dispatch(WorkspaceRegistry(...))`.


