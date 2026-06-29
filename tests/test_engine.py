"""Mocked tests for the end-to-end engine."""

from __future__ import annotations

from typing import Any

from ledgerlens.engine import answer_question
from ledgerlens.solver.schema import Operand, ProgramProposal


class _Stub:
    def __init__(self, proposal: ProgramProposal) -> None:
        self._proposal = proposal

    def chat_structured(self, messages: Any, response_model: Any, **kwargs: Any) -> ProgramProposal:
        return self._proposal


def test_answer_question_accepts_and_traces_steps() -> None:
    proposal = ProgramProposal(
        reasoning="r",
        operands=[Operand(name="net revenue 2023", value="5829", citation="row")],
        program="subtract(5829, 5735), divide(#0, 5735)",
    )
    answer = answer_question(_Stub(proposal), "percent change?", "net revenue 5829 and 5735")
    assert answer.answered
    assert answer.answer is not None
    assert [s.op for s in answer.steps] == ["subtract", "divide"]
    assert answer.steps[1].args == ["#0", "5735"]


def test_answer_question_abstains_on_fabricated_operand() -> None:
    proposal = ProgramProposal(reasoning="r", operands=[], program="subtract(9999, 5735)")
    answer = answer_question(_Stub(proposal), "q?", "net revenue 5829 and 5735")
    assert not answer.answered
    assert answer.answer is None
    assert any("ungrounded" in reason for reason in answer.abstain_reasons)
