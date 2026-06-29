"""The agent's verified-computation tool: retrieve evidence, then compute via LedgerLens.

This is the *only* source of numbers the agent has — so the agent cannot fabricate a figure.
Every value comes back from the gated engine, with its citations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ledgerlens.engine import answer_question
from ledgerlens.llm import LLMClient

# Given a question and a top-k, return the evidence text to reason over.
RetrieveFn = Callable[[str, int], str]


@dataclass
class Finding:
    """A single verified (or abstained) computation."""

    question: str
    answered: bool
    answer: str | None
    citations: list[str] = field(default_factory=list)
    abstain_reasons: list[str] = field(default_factory=list)


class VerifiedCalculator:
    """Retrieve evidence for a sub-question, then compute the answer through LedgerLens."""

    def __init__(self, client: LLMClient, retrieve: RetrieveFn) -> None:
        self.client = client
        self.retrieve = retrieve

    def compute(self, question: str, *, k: int = 8) -> Finding:
        context = self.retrieve(question, k)
        result = answer_question(self.client, question, context)
        return Finding(
            question=question,
            answered=result.answered,
            answer=result.answer,
            citations=[op.citation for op in result.operands],
            abstain_reasons=result.abstain_reasons,
        )
