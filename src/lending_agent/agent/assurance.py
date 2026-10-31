"""
Phase 6 -- Node 4 of the pipeline: verify_grounding (the assembling part).

Takes the Phase 5 result and the evidence it was based on, and returns a
VerifiedResult in which every finding carries:
  - its grounding check (see grounding.py),
  - whether a human needs to look at it, and why,
  - whether it conflicts with another finding.

Findings that fail the checks are KEPT and flagged, never dropped: a human
reviewer should see exactly what the model claimed and why it was doubted.

CONFLICT RULE. Two findings conflict when they describe the same limit
(same kind, same minimum/maximum or date direction, same unit, same
classification, same scope) but give different values -- in two documents, or
in two places in one document. Findings with *different* scopes, such as
"LTV below 90 percent" versus "LTV above 90 percent", are tiers, not conflicts.
Only grounded findings are compared, so a misread finding cannot create a
false conflict. Scope wording comes from the model and can vary, so a human
should still glance at several values listed for the same field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from lending_agent.agent.extraction import (
    KIND_NUMERIC,
    STATUS_FOUND,
    ExtractionResult,
    Finding,
)
from lending_agent.agent.grounding import GroundingCheck, check_finding
from lending_agent.agent.retrieval import Evidence

_SCOPE_STOPWORDS = {"the", "a", "an", "of", "for", "to", "with", "and", "or", "in", "on", "at", "by", "is", "are"}


@dataclass
class VerifiedFinding:
    finding: Finding
    check: GroundingCheck
    conflicts_with: list[int] = field(default_factory=list)   # 1-based positions of the other findings

    @property
    def grounded(self) -> bool:
        return self.check.grounded

    @property
    def review_reasons(self) -> list[str]:
        reasons = list(self.check.warnings)
        if self.conflicts_with:
            others = ", ".join(f"#{i}" for i in self.conflicts_with)
            reasons.append(f"conflicts with finding {others}: a different value for the same limit")
        return reasons

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def to_dict(self) -> dict:
        return {
            **self.finding.to_dict(),
            "grounded": self.grounded,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "checks": self.check.to_dict(),
            "conflicts_with": self.conflicts_with,
        }


@dataclass(frozen=True)
class Conflict:
    kind: str
    limit: str                      # "maximum", "minimum" or the date direction
    unit: str | None
    scope: str | None
    classification: str
    values: list[dict]              # each: {"finding": n, "value": ..., "source_file": ..., "page_number": ...}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "limit": self.limit, "unit": self.unit, "scope": self.scope,
            "classification": self.classification, "values": self.values,
        }


@dataclass
class VerifiedResult:
    field: str
    status: str
    findings: list[VerifiedFinding] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def verified(self) -> bool:
        """True only when there are findings and every one passed the hard checks."""
        return self.status == STATUS_FOUND and bool(self.findings) and all(f.grounded for f in self.findings)

    @property
    def needs_review(self) -> bool:
        """True when a person should look: a finding failed a check, or values conflict."""
        return any(f.needs_review for f in self.findings) or bool(self.conflicts)

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "status": self.status,
            "verified": self.verified,
            "needs_review": self.needs_review,
            "findings": [f.to_dict() for f in self.findings],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "rejected": self.rejected,
            "error": self.error,
        }


def _normalise_scope(scope: str | None) -> str:
    words = re.findall(r"[a-z0-9]+", (scope or "").lower())
    return " ".join(w for w in words if w not in _SCOPE_STOPWORDS)


def _limit_and_value(f: Finding) -> tuple[str, object]:
    if f.kind == KIND_NUMERIC:
        return f.bound_type or "", f.value
    return f.date_direction or "", f.date


def detect_conflicts(findings: Sequence[VerifiedFinding]) -> list[Conflict]:
    """Group grounded findings that describe the same limit; report groups holding different values.
    Also records, on each finding involved, which other findings it conflicts with."""
    groups: dict[tuple, list[tuple[int, VerifiedFinding]]] = {}
    for position, vf in enumerate(findings, 1):
        if not vf.grounded:
            continue
        f = vf.finding
        limit, _ = _limit_and_value(f)
        key = (f.kind, limit, f.unit, f.classification, _normalise_scope(f.applies_to))
        groups.setdefault(key, []).append((position, vf))

    conflicts: list[Conflict] = []
    for (kind, limit, unit, classification, scope), members in groups.items():
        distinct = {_limit_and_value(vf.finding)[1] for _, vf in members}
        if len(distinct) < 2:
            continue
        for position, vf in members:
            vf.conflicts_with = [p for p, other in members if p != position
                                 and _limit_and_value(other.finding)[1] != _limit_and_value(vf.finding)[1]]
        conflicts.append(
            Conflict(
                kind=kind, limit=limit, unit=unit, scope=scope or None, classification=classification,
                values=[
                    {
                        "finding": position,
                        "value": _limit_and_value(vf.finding)[1],
                        "source_file": vf.finding.source_file,
                        "page_number": vf.finding.page_number,
                    }
                    for position, vf in members
                ],
            )
        )
    return conflicts


def verify_extraction(result: ExtractionResult, evidence: Sequence[Evidence]) -> VerifiedResult:
    """Check every finding in `result` against the evidence it cites, then look for conflicts."""
    text_by_chunk = {e.chunk_id: e.text for e in evidence}
    verified = [
        VerifiedFinding(finding=f, check=check_finding(f, text_by_chunk.get(f.chunk_id, "")))
        for f in result.findings
    ]
    conflicts = detect_conflicts(verified)
    return VerifiedResult(
        field=result.field,
        status=result.status,
        findings=verified,
        conflicts=conflicts,
        rejected=list(result.rejected),
        error=result.error,
    )
