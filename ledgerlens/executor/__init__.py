"""Deterministic program-DSL executor (P1) — the crown jewel.

Runs the symbolic reasoning program proposed by the LLM (operands + ops). Pure
Python, total (returns a value-or-error result), and unit-agnostic. The LLM never
performs arithmetic here.
"""
