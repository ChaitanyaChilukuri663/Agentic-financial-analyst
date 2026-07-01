# Agent eval

- **Answer accuracy (answerable):** 19/19 (100%)
  - lookup: 9/9 (100%)
  - growth: 7/7 (100%)
  - compare: 3/3 (100%)
- **Abstention correctness:** 4/4 (100%)
- **Faithfulness:** 36/36 numbers in answers trace to a verified figure (100%)

| # | category | question | correct | faithful | answer |
|---|---|---|---|---|---|
| 1 | lookup | What was Apple's (AAPL) revenue in its most recent fiscal... | ✅ | ✅ | Apple's revenue in its most recent fiscal year FY2025 was $416,161,... |
| 2 | lookup | What was Microsoft's (MSFT) revenue in its most recent fi... | ✅ | ✅ | Microsoft's revenue in its most recent fiscal year FY2025 was $281,... |
| 3 | lookup | What was NVIDIA's (NVDA) revenue in its most recent fisca... | ✅ | ✅ | NVIDIA's revenue in its most recent fiscal year FY2026 was 215,938,... |
| 4 | lookup | What was Apple's (AAPL) net income in its most recent fis... | ✅ | ✅ | Apple's net income in its most recent fiscal year FY2025 was 112,01... |
| 5 | lookup | What was Microsoft's (MSFT) net income in its most recent... | ✅ | ✅ | Microsoft's net income in its most recent fiscal year FY2025 was 10... |
| 6 | lookup | What was NVIDIA's (NVDA) net income in its most recent fi... | ✅ | ✅ | NVIDIA's net income in its most recent fiscal year FY2026 was 120,0... |
| 7 | lookup | What was Apple's (AAPL) gross profit in its most recent f... | ✅ | ✅ | Apple's gross profit in its most recent fiscal year (FY2025) was 19... |
| 8 | lookup | What was Apple's (AAPL) revenue in fiscal year 2023? | ✅ | ✅ | Apple's revenue in fiscal year 2023 was $383,285,000,000, according... |
| 9 | lookup | What was Microsoft's (MSFT) revenue in fiscal year 2024? | ✅ | ✅ | Microsoft's revenue in fiscal year 2024 was 245,122,000,000 USD, ac... |
| 10 | growth | How fast did Apple's (AAPL) revenue grow in its most rece... | ✅ | ✅ | Apple's revenue grew from $391.0 billion in FY2024 to $416.2 billio... |
| 11 | growth | How fast did Microsoft's (MSFT) revenue grow in its most ... | ✅ | ✅ | Microsoft's revenue grew by approximately 14.93% in its most recent... |
| 12 | growth | How fast did NVIDIA's (NVDA) revenue grow in its most rec... | ✅ | ✅ | NVIDIA's revenue grew by approximately 65.47% in its most recent fi... |
| 13 | growth | How fast did Apple's (AAPL) net income grow in its most r... | ✅ | ✅ | Apple's net income grew by approximately 19.5% in its most recent f... |
| 14 | growth | How fast did Microsoft's (MSFT) net income grow in its mo... | ✅ | ✅ | Microsoft's net income grew from $88.136 billion in FY2024 to $101.... |
| 15 | growth | How fast did NVIDIA's (NVDA) net income grow in its most ... | ✅ | ✅ | NVIDIA's net income grew from $72.88 billion in FY2025 to $120.07 b... |
| 16 | growth | How fast did Microsoft's (MSFT) gross profit grow in its ... | ✅ | ✅ | Microsoft's gross profit grew from $171.01 billion in FY2024 to $19... |
| 17 | compare | Which grew revenue faster last fiscal year, Apple (AAPL) ... | ✅ | ✅ | Microsoft's revenue grew by approximately 14.93% last fiscal year, ... |
| 18 | compare | Which grew revenue faster last fiscal year, Apple (AAPL) ... | ✅ | ✅ | Apple's revenue grew by approximately 6.43% from FY2024 to FY2025, ... |
| 19 | compare | Which grew revenue faster last fiscal year, Microsoft (MS... | ✅ | ✅ | Microsoft's revenue grew by approximately 14.93% from FY2024 to FY2... |
| 20 | abstain | What was Apple's (AAPL) inventory in its most recent fisc... | ✅ | ✅ | I couldn't find the figures needed to answer this in the company's ... |
| 21 | abstain | What is Microsoft's (MSFT) total employee headcount? | ✅ | ✅ | I attempted to find Microsoft's total employee headcount using XBRL... |
| 22 | abstain | What was NVIDIA's (NVDA) advertising expense last fiscal ... | ✅ | ✅ | I couldn't find the figures needed to answer this in the company's ... |
| 23 | abstain | What is Apple's (AAPL) customer satisfaction score? | ✅ | ✅ | Apple does not report a specific customer satisfaction score in its... |
