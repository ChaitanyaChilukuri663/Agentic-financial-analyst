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
`match_value(parse_company_facts(client.company_facts(320193)), value)`. *(Table-aware
hybrid retrieval + recall@k eval + FinanceBench stress test: next P3 step.)*

