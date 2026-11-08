"""
Phase 8 -- turns the agent's answer (Phase 7) into the JSON a chat client or web page shows.

Plain code, no LLM: the sentence a person reads is assembled only from findings that already
passed Phase 6, so the wording cannot add anything the policy did not say.

Headline values (`min_value`, `max_value`, `cutoff_date`) are taken ONLY from findings that are
grounded AND tagged firm_rule, and only when those findings agree. Tiers (72 months below 90
percent LTV, 60 above) have no single headline value: it is left empty and every value is listed.

`route` is the agent's route, with one extra rule: if the agent found values but none is a
grounded firm rule (only examples or guidance), the route becomes "needs_review", because a
tester must not use an illustration as a hard limit. The agent's original route is kept in
`agent_route`.
"""
from __future__ import annotations

from typing import Any, Sequence

from lending_agent.agent.assurance import VerifiedFinding
from lending_agent.agent.extraction import KIND_DATE, KIND_NUMERIC, STATUS_FOUND

ROUTE_VERIFIED, ROUTE_NEEDS_REVIEW, ROUTE_NOT_FOUND, ROUTE_ERROR = "verified", "needs_review", "not_found", "error"

_DIRECTION_WORDS = {"on_or_after": "on or after", "after": "after", "on_or_before": "on or before", "before": "before"}


def _number(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else f"{value:g}"


def _describe(vf: VerifiedFinding) -> str:
    f = vf.finding
    if f.kind == KIND_NUMERIC:
        text = f"{f.bound_type} {_number(f.value)}" + (f" {f.unit}" if f.unit else "")
    else:
        text = f"a cutoff date, {_DIRECTION_WORDS.get(f.date_direction or '', f.date_direction)} {f.date}"
        if getattr(f, "date_precision", None) == "month":
            text += " (month only, day not stated)"
    if f.applies_to:
        text += f" (applies to: {f.applies_to})"
    return text


def _cite(vf: VerifiedFinding) -> str:
    f = vf.finding
    return f"{f.source_file}, page {f.page_number}" if f.page_number else f"{f.source_file}"


def _headline(firm: Sequence[VerifiedFinding]) -> dict:
    out: dict[str, Any] = {}

    def agreed(candidates, key):
        return candidates[0] if candidates and len({key(c.finding) for c in candidates}) == 1 else None

    numeric = [v for v in firm if v.finding.kind == KIND_NUMERIC]
    units: set = set()
    for bound, name in (("minimum", "min_value"), ("maximum", "max_value")):
        chosen = agreed([v for v in numeric if v.finding.bound_type == bound], lambda f: (f.value, f.unit))
        if chosen:
            out[name] = chosen.finding.value
            units.add(chosen.finding.unit)
            out.setdefault("_source", chosen)
    if len(units) == 1:
        out["unit"] = next(iter(units))

    chosen = agreed([v for v in firm if v.finding.kind == KIND_DATE], lambda f: (f.date, f.date_direction))
    if chosen:
        out.update(cutoff_date=chosen.finding.date, date_direction=chosen.finding.date_direction,
                   date_precision=getattr(chosen.finding, "date_precision", None))
        out.setdefault("_source", chosen)
    return out


def build_payload(answer: Any) -> dict:
    """`answer` is a Phase 7 AgentAnswer: .route .summary .result (a VerifiedResult) .attempts .widened .trace"""
    result = answer.result
    base: dict[str, Any] = {
        "field": result.field, "agent_route": answer.route, "route": answer.route,
        "grounded": False, "needs_review": False, "review_reasons": [],
        "min_value": None, "max_value": None, "unit": None,
        "cutoff_date": None, "date_direction": None, "date_precision": None,
        "source_quote": None, "source_file": None, "page_number": None,
        "findings": [], "conflicts": [], "rejected": list(result.rejected), "error": result.error,
        "attempts": answer.attempts, "widened_search": answer.widened, "trace": list(answer.trace),
        "search": dict(getattr(answer, "search", None) or {}),
    }

    if answer.route == ROUTE_ERROR:
        base["answer"] = f"I could not answer this: {result.error}"
        return base

    if answer.route == ROUTE_NOT_FOUND or result.status != STATUS_FOUND:
        label = result.field or "this field"
        base["answer"] = (f"I did not find a boundary or date cutoff for {label} in the paragraphs I read. "
                          "That does not prove the policy has none: it may be worded without a number or date, "
                          "or sit in a paragraph that was not retrieved.")
        return base

    findings = result.findings
    firm = [v for v in findings if v.grounded and v.finding.classification == "firm_rule"]
    soft = [v for v in findings if v.grounded and v.finding.classification != "firm_rule"]
    head = _headline(firm)
    source = head.pop("_source", None)

    reasons: list[str] = []
    for position, vf in enumerate(findings, 1):
        reasons.extend(f"finding #{position}: {r}" for r in vf.review_reasons)
    if not firm:
        reasons.append("no firm rule was confirmed: only example or guidance wording was found, "
                       "or no finding passed the grounding checks")
    needs_review = answer.route == ROUTE_NEEDS_REVIEW or result.needs_review or not firm

    label = result.field
    if firm:
        lines = [f"- {_describe(v)}, per {_cite(v)}" for v in firm]
        text = f"For {label}: {lines[0][2:]}." if len(firm) == 1 else (
            f"For {label}, the policy gives these limits:\n" + "\n".join(lines)
            + ("" if head else "\nThey differ by scope or by document, so there is no single headline value."))
        if soft:
            text += "\nAlso found, but not firm rules: " + "; ".join(_describe(v) for v in soft) + "."
    else:
        text = (f"For {label} I found only {'example or guidance' if soft else 'unconfirmed'} wording, not a firm rule. "
                "Do not use these as hard limits without checking the source.")
    if result.conflicts:
        text += "\nWarning: the policy gives different values for the same limit. A person should confirm which applies."
    if needs_review:
        text += "\nA person should review this before it is used: see review_reasons."

    base.update(head)
    base.update(
        answer=text, route=ROUTE_NEEDS_REVIEW if needs_review else ROUTE_VERIFIED,
        grounded=result.verified, needs_review=needs_review, review_reasons=reasons,
        source_quote=source.finding.quote if source else None,
        source_file=source.finding.source_file if source else None,
        page_number=source.finding.page_number if source else None,
        findings=[v.to_dict() for v in findings], conflicts=[c.to_dict() for c in result.conflicts],
    )
    return base
