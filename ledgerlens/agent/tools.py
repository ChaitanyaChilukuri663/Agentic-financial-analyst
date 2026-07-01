"""The agent's tools — its only ways to get a figure or compute over figures.

- ``xbrl_value``: ONE exact figure straight from the company's filed XBRL data (ground truth;
  latest fiscal year by default).
- ``compute``: deterministic arithmetic (via the P1 executor) over figures the agent ALREADY
  obtained — every input must trace to a previously verified figure, so the agent cannot slip
  in a fabricated number.
- ``passage``: a short text snippet for a qualitative / non-numeric fact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ledgerlens.agent.schema import AgentAction
from ledgerlens.agent.workspace import FilingWorkspace, WorkspaceRegistry
from ledgerlens.executor.dsl import NumberArg, Op, Program, RefArg, Step
from ledgerlens.executor.program import execute

Dispatch = Callable[[AgentAction], "ToolResult"]
_REL_TOL = Decimal("0.001")


@dataclass
class ToolResult:
    """The outcome of one tool call."""

    tool: str
    ok: bool
    output: str
    citation: str = ""
    note: str = ""


# Common financial terms -> canonical us-gaap concept(s), tried in priority order. This keeps
# 'revenue' from matching backlog/segment lines like RevenueRemainingPerformanceObligation.
_REVENUE = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "net sales": _REVENUE,
    "revenue": _REVENUE,
    "sales": _REVENUE,
    "net income": ("NetIncomeLoss",),
    "gross profit": ("GrossProfit",),
    "operating income": ("OperatingIncomeLoss",),
    "research and development": ("ResearchAndDevelopmentExpense",),
    "total assets": ("Assets",),
    "assets": ("Assets",),
    "total liabilities": ("Liabilities",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
}


def _resolve_concepts(keyword: str) -> tuple[str, ...]:
    kw = keyword.lower().strip()
    for alias in sorted(_CONCEPT_ALIASES, key=len, reverse=True):  # longest alias first
        if alias in kw:
            return _CONCEPT_ALIASES[alias]
    return ()


def tool_xbrl_value(
    workspace: FilingWorkspace, keyword: str, fiscal_year: int | None
) -> ToolResult:
    # Pool facts across ALL alias concepts, not just the first that matches: a company may
    # report a metric under different us-gaap concepts across years (e.g. NVDA's recent revenue
    # is under `Revenues` while older years sit under the ASC-606 concept). Pooling lets the
    # latest-year + max-abs selection below pick the truly most recent figure.
    candidates: list = []
    for concept in _resolve_concepts(keyword):
        candidates.extend(f for f in workspace.facts if f.concept == concept)
    if not candidates:  # fallback: substring match on the concept name
        kw = keyword.lower().strip()
        candidates = [f for f in workspace.facts if kw in f.concept.lower()]
    if fiscal_year is not None:
        candidates = [f for f in candidates if f.fiscal_year == fiscal_year]
    annual = [f for f in candidates if f.fiscal_period == "FY"] or candidates
    ten_k = [f for f in annual if f.form == "10-K"] or annual
    if not ten_k:
        return ToolResult(
            "xbrl_value", False, "", note=f"no XBRL fact for {keyword!r} FY{fiscal_year}"
        )
    if fiscal_year is None:
        latest = max((f.fiscal_year or 0) for f in ten_k)
        ten_k = [f for f in ten_k if (f.fiscal_year or 0) == latest]
    fact = max(ten_k, key=lambda f: abs(f.value))  # consolidated total for that concept/year
    return ToolResult(
        "xbrl_value", True, str(fact.value), f"{fact.concept} FY{fact.fiscal_year} ({fact.unit})"
    )


def tool_passage(workspace: FilingWorkspace, query: str) -> ToolResult:
    text = workspace.retrieve(query, 3).strip()
    if not text:
        return ToolResult("passage", False, "", note="no passage found")
    return ToolResult("passage", True, text[:600], "filing text")


def tool_compute(
    operation: str, values: list[float], ledger: list[Decimal], *, rel_tol: Decimal = _REL_TOL
) -> ToolResult:
    decimals: list[Decimal] = []
    for value in values:
        dec = _to_decimal(value)
        if dec is None:
            return ToolResult("compute", False, "", note=f"non-numeric value {value!r}")
        decimals.append(dec)
    if not decimals:
        return ToolResult("compute", False, "", note="no values provided")
    for dec in decimals:
        if not _in_ledger(dec, ledger, rel_tol):
            note = f"{dec} is not a previously verified figure"
            return ToolResult("compute", False, "", note=note)
    program = _program_for(operation, decimals)
    if program is None:
        return ToolResult("compute", False, "", note=f"unsupported operation {operation!r}")
    result = execute(program)
    if not result.ok:
        note = result.error.code if result.error else "failed"
        return ToolResult("compute", False, "", note=note)
    inputs = ", ".join(str(d) for d in decimals)
    return ToolResult("compute", True, str(result.value), f"{operation} of {inputs}")


def make_dispatch(registry: WorkspaceRegistry) -> Dispatch:
    """Build the dispatcher the agent calls; it keeps a ledger of verified figures."""
    ledger: list[Decimal] = []

    def dispatch(action: AgentAction) -> ToolResult:
        if action.tool == "compute":
            return tool_compute(action.operation, action.values, ledger)
        workspace = registry.get(action.company)
        if workspace is None:
            return ToolResult(action.tool, False, "", note=f"unknown company {action.company!r}")
        if action.tool == "xbrl_value":
            result = tool_xbrl_value(workspace, action.query, action.fiscal_year)
            if result.ok:
                dec = _to_decimal(result.output)
                if dec is not None:
                    ledger.append(dec)
            return result
        if action.tool == "passage":
            return tool_passage(workspace, action.query)
        return ToolResult(action.tool, False, "", note=f"unknown tool {action.tool!r}")

    return dispatch


def _to_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _in_ledger(value: Decimal, ledger: list[Decimal], tol: Decimal) -> bool:
    return any(
        abs(value - known) <= tol * max(abs(value), abs(known), Decimal(1)) for known in ledger
    )


def _program_for(operation: str, decimals: list[Decimal]) -> Program | None:
    def num(x: Decimal) -> NumberArg:
        return NumberArg(value=x)

    if operation == "difference" and len(decimals) == 2:
        return Program(steps=[Step(op=Op.SUBTRACT, args=[num(decimals[0]), num(decimals[1])])])
    if operation == "ratio" and len(decimals) == 2:
        return Program(steps=[Step(op=Op.DIVIDE, args=[num(decimals[0]), num(decimals[1])])])
    if operation == "percent_change" and len(decimals) == 2:
        return Program(
            steps=[
                Step(op=Op.SUBTRACT, args=[num(decimals[1]), num(decimals[0])]),
                Step(op=Op.DIVIDE, args=[RefArg(step=0), num(decimals[0])]),
            ]
        )
    if operation in ("sum", "average") and decimals:
        second = decimals[1] if len(decimals) > 1 else Decimal(0)
        steps = [Step(op=Op.ADD, args=[num(decimals[0]), num(second)])]
        for dec in decimals[2:]:
            steps.append(Step(op=Op.ADD, args=[RefArg(step=len(steps) - 1), num(dec)]))
        if operation == "average":
            divisor = num(Decimal(len(decimals)))
            steps.append(Step(op=Op.DIVIDE, args=[RefArg(step=len(steps) - 1), divisor]))
        return Program(steps=steps)
    return None
