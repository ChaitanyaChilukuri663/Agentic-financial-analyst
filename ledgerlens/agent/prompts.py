"""Prompts for the research agent's plan and synthesis steps."""

from __future__ import annotations

_PLAN_SYSTEM = """You are a financial research analyst. Break the user's task into a short list
of specific sub-questions, each answerable from a company's filing and each asking for ONE
calculation that requires arithmetic — a change, a growth rate, a ratio, or a comparison
between periods. Avoid plain value look-ups (the calculator needs an operation). Do not answer
them. Prefer 2-3 focused sub-questions."""

_SYNTH_SYSTEM = """You are a financial analyst writing the final answer. You are given VERIFIED
findings — each a sub-question, its computed answer, and a citation. Write a concise summary
that answers the task and cites the figures, then state the trend.

CRITICAL: use ONLY the provided computed figures. Never invent, estimate, or recompute a
number. If a finding was not answered, say the data was insufficient for that part."""


def build_plan_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user", "content": task},
    ]


def build_synth_messages(task: str, findings_text: str) -> list[dict[str, str]]:
    content = f"Task: {task}\n\nVerified findings:\n{findings_text}"
    return [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user", "content": content},
    ]
