"""Shared fakes for the Phase 8 tests. No Ollama, no Chroma, no web framework."""
from __future__ import annotations

from dataclasses import dataclass, field

from lending_agent.agent.assurance import verify_extraction
from lending_agent.agent.extraction import STATUS_ERROR, STATUS_FOUND, STATUS_NOT_FOUND, ExtractionResult, Finding
from lending_agent.agent.retrieval import Evidence

TENURE = "The repayment period shall not extend beyond thirty (30) years from disbursement."
LTV_HIGH = "Loans with LTV ratios less than 90 percent may have loan terms up to 72 months."
LTV_LOW = "Loans with LTV ratios greater than 90 percent are limited to 60 months."
EXAMPLE = "For example, some banks allow loan terms up to 72 months."


def evidence(chunk_id, text, source="policy.pdf", page=5):
    return Evidence(chunk_id=chunk_id, text=text, source_file=source, page_number=page, doc_type="pdf", best_distance=0.2)


def numeric(chunk_id, quote, value, bound="maximum", unit="years", classification="firm_rule", applies_to=None,
            source="policy.pdf", page=5):
    return Finding(kind="numeric_boundary", classification=classification, quote=quote, source_file=source,
                   page_number=page, chunk_id=chunk_id, bound_type=bound, value=value, unit=unit, applies_to=applies_to)


@dataclass
class FakeAnswer:
    """Looks like the Phase 7 AgentAnswer (which needs langgraph to import)."""
    field: str
    route: str
    summary: str
    result: object
    attempts: int = 1
    widened: bool = False
    trace: list = field(default_factory=list)


def answer_for(findings, evidence_list, field_name="loan_tenure", route=None):
    extraction = ExtractionResult(field=field_name, status=STATUS_FOUND, findings=list(findings))
    verified = verify_extraction(extraction, evidence_list)
    if route is None:
        route = "verified" if verified.verified and not verified.needs_review else "needs_review"
    return FakeAnswer(field=field_name, route=route, summary="", result=verified)


def not_found_answer(field_name="loan_tenure"):
    verified = verify_extraction(ExtractionResult(field=field_name, status=STATUS_NOT_FOUND), [])
    return FakeAnswer(field=field_name, route="not_found", summary="", result=verified)


def error_answer(message="Ollama is down", field_name="loan_tenure"):
    verified = verify_extraction(ExtractionResult(field=field_name, status=STATUS_ERROR, error=message), [])
    return FakeAnswer(field=field_name, route="error", summary="", result=verified)


class ScriptedLLM:
    """Hands out one reply per call, in order. An Exception instance is raised instead of returned."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeStore:
    def __init__(self):
        self.chunks, self.calls = [], []

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)
        return len(chunks)

    def delete_source(self, name):
        self.calls.append(("delete", name))
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source_file != name]
        return before - len(self.chunks)

    def sources(self):
        counts = {}
        for c in self.chunks:
            counts[c.source_file] = counts.get(c.source_file, 0) + 1
        return dict(sorted(counts.items()))
