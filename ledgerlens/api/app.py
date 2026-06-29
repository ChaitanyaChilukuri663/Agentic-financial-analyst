"""Thin FastAPI service for LedgerLens.

    uvicorn ledgerlens.api.app:app --reload

POST /answer with a question + evidence; returns the verified answer, the program, the
computed steps, the cited operands, and any abstention reasons.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ledgerlens.engine import answer_question
from ledgerlens.llm import LLMClient

app = FastAPI(title="LedgerLens", version="0.1.0")
client = LLMClient()


class AnswerRequest(BaseModel):
    """A question to answer over a block of financial evidence."""

    question: str = Field(description="The financial question to answer.")
    context: str = Field(description="Evidence (filing text / table rows) to reason over.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    result = answer_question(client, request.question, request.context)
    return {
        "question": result.question,
        "answered": result.answered,
        "answer": result.answer,
        "abstain_reasons": result.abstain_reasons,
        "program": result.program,
        "steps": [asdict(step) for step in result.steps],
        "operands": [operand.model_dump() for operand in result.operands],
    }
