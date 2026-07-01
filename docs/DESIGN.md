# Design notes

Why this system is built the way it is. Each decision lists the reasoning and the trade-off it
accepts — the point is to show the *judgment*, not just the code.

---

### 1. The LLM proposes a program; a deterministic engine does the math

**Decision.** The model never emits a final number. It emits a small reasoning *program*
(`subtract(5829, 5735), divide(#0, 5735)`); a pure-Python executor runs it.

**Why.** Language models are unreliable calculators — they miscompute chained arithmetic and
state numbers that merely *look* right. Separating *what to compute* (a reasoning task, which
the model is good at) from *doing the computation* (a mechanical task, which code is good at)
removes a whole class of errors. This is the published *Program-of-Thoughts / PAL* idea; the
executor here reproduces **99.5% of 8,281 expert FinQA programs with zero LLM involvement**, so
the arithmetic layer is effectively solved.

**Trade-off.** The model must express intent in a constrained DSL, and a question the DSL can't
express can't be answered. That's an acceptable price: a narrower, verifiable surface beats a
wider, unverifiable one.

### 2. Abstain on *trustworthy* signals, not on the model's self-confidence

**Decision.** The accept/abstain gate keys on grounding (does every operand trace to the
filing or filed XBRL?), program validity, and numeric sanity — never on the model saying "I'm
confident."

**Why.** A model's stated confidence is uncorrelated with correctness; it is confident when it
is wrong. Grounding and validity are *checkable properties of the world*, so they're a sound
basis for refusing to answer. The gates lift answer precision **60% → 64.5% with 0%
false-abstentions** on FinQA.

**Trade-off.** The system declines some answerable questions rather than risk a wrong one. For a
finance tool, a known "I can't verify this" is far more valuable than a confident guess.

### 3. Ground every figure in XBRL, dated by its period-end — not the filing's fiscal year

**Decision.** Numeric answers resolve to the company's filed XBRL facts. Each fact is dated by
the **period-end date of the value**, not by SEC's `fy` field.

**Why.** XBRL is the machine-readable, audited source of truth — the strongest possible anchor
for "did the company actually report this number?" But SEC's `fy` tags each fact with the
*filing's* fiscal year, so a 10-K's three comparative years collapse to one year and dedup picks
the wrong figure (this really happened: Apple's FY2023/FY2024 revenue both showed FY2022's
$394B). Dating by the period end fixes it. See `ingest/xbrl.py::_fiscal_year`.

**Trade-off.** Companies with unusual fiscal-year labelling (some retailers) could still be
mislabelled; the current demo companies (Sept/June/Jan year-ends) are all correct.

### 4. The agent discovers the latest year from the data — it never guesses

**Decision.** For "most recent year" questions the agent omits the fiscal year on its first
lookup (the tool returns the latest value *and its year*), then works backwards for the prior
year.

**Why.** The model's training cutoff makes it hard-code stale years (it thought Apple's latest
year was FY2023 and computed growth for the wrong period). The filed data is newer than the
model "knows," so the source — not the model's memory — must decide what "latest" means.

**Trade-off.** Costs one extra tool round-trip, in exchange for always tracking the newest
filing without prompt edits.

### 5. No LangChain — the agent loop is hand-rolled

**Decision.** The ReAct loop (plan → act → observe → revise), tool dispatch, loop-guarding, and
telemetry are ~130 lines of explicit Python.

**Why.** For a system whose whole selling point is *verifiability*, an opaque framework works
against the goal. Hand-rolling keeps every decision inspectable, makes the "can't fabricate"
guarantee auditable end-to-end, and demonstrates understanding of *how* an agent loop actually
works rather than which library to import.

**Trade-off.** More code to own; no free ecosystem integrations. Worth it for a portfolio piece
about correctness.

### 6. Hybrid retrieval (BM25 + dense + Reciprocal Rank Fusion)

**Decision.** Retrieve evidence with lexical BM25 and dense embeddings, fused with RRF.

**Why.** Financial text mixes exact tokens (line-item names, "diluted EPS") where BM25 wins with
paraphrase where dense wins. Fusing both beats either alone: recall@5 **85.6%** vs 81.9% (BM25)
and 82.6% (dense).

**Trade-off.** Two indexes to build and maintain instead of one.

### 7. The hosted demo serves committed data bundles

**Decision.** Six companies (AAPL, MSFT, NVDA, GOOGL, AMZN, META) ship as small committed JSON
bundles; the hosted app answers them with **zero live SEC calls**.

**Why.** SEC EDGAR blocks shared cloud/datacenter IPs (Streamlit Cloud gets a 403), so a
live-fetch demo is dead on arrival for reviewers. Bundling the exact figures the tools need
(~15 concepts, latest chunks) keeps the demo fast, offline, and reproducible. Local runs still
fetch any ticker live.

**Trade-off.** The hosted demo is limited to the bundled companies and metrics; anything else
correctly abstains.

### 8. Evaluation is the centerpiece, not an afterthought

**Decision.** Every headline claim has a runnable eval: executor accuracy, determinism payoff,
retrieval recall, abstention precision, and an **agent benchmark** (accuracy / faithfulness /
abstention) — see [`evals/`](../evals). The agent eval measures faithfulness directly: **every
number in every answer traced to a verified figure (100%)**.

**Why.** Claims about "trustworthy AI" mean nothing without measurement. Measuring faithfulness
also told us empirically that a separate faithfulness *guard* wasn't yet needed — a decision made
from data, not assumption.

**Trade-off.** The agent eval needs a live LLM (a few cents to run) and is not part of the
offline test suite.
