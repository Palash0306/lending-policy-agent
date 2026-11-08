"""
Phase 7 -- the pipeline as a LangGraph graph.

    START -> expand -> retrieve -> extract -> verify -> (decide) -> respond -> END
                          ^           ^                    |
                          |           +------ retry -------+   extraction failed or a finding
                          +------- widen ------------------+   was not grounded: ask again with feedback
                                                               nothing found: search wider, once

The steps themselves are the ones built in Phases 4-6; this file only decides what
happens between them. That is what makes it an agent rather than a fixed script:

  * RETRY  - if the model's reply was not valid JSON, or some findings failed the
             grounding check, extraction runs again (at most `max_attempts` times in
             total) with the problems pointed out in the prompt. Plain re-running would be
             pointless: the model runs at temperature 0 and would repeat itself.
  * WIDEN  - if nothing was found, the search is repeated once with a larger `top_k`.
  * ROUTE  - the final answer is routed to one of four outcomes:
             "verified", "needs_review" (a person must look), "not_found", "error".
             Ollama being unreachable is "error" and is never retried or called "not found".

Every node is timed; the list is returned as `trace`, ready for the run report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from lending_agent.agent.assurance import VerifiedResult, verify_extraction
from lending_agent.agent.extraction import (
    ALL_KINDS, KIND_NUMERIC, STATUS_ERROR, STATUS_NOT_FOUND, ExtractionResult, extract_conditions,
)
from lending_agent.agent.field_expansion import FieldExpansion, expand_field_terms
from lending_agent.agent.retrieval import Evidence, retrieve_evidence

ROUTE_VERIFIED = "verified"
ROUTE_NEEDS_REVIEW = "needs_review"
ROUTE_NOT_FOUND = "not_found"
ROUTE_ERROR = "error"

_RETRYABLE_ERRORS = ("not valid JSON", "no 'findings' list")  # a flaky reply, not a dead server


@dataclass
class AgentConfig:
    kinds: tuple[str, ...] = ALL_KINDS
    top_k: int = 6
    widen_factor: int = 2          # on "nothing found", search again with top_k * this (1 turns widening off)
    max_attempts: int = 2          # total extraction attempts, including the first
    num_ctx: int = 4096
    focus: bool = True
    use_llm_expansion: bool = True
    source_file: str | None = None


class AgentState(TypedDict, total=False):
    query: str
    expansion: FieldExpansion
    top_k: int
    evidence: list[Evidence]
    extraction: ExtractionResult
    attempt: int
    widened: bool
    feedback: str
    verified: VerifiedResult
    answer: "AgentAnswer"
    trace: list[dict]


@dataclass
class AgentAnswer:
    field: str
    route: str
    summary: str
    result: VerifiedResult
    attempts: int
    widened: bool
    trace: list[dict] = field(default_factory=list)
    search: dict = field(default_factory=dict)       # what was searched and read: terms, paragraphs, how many were sent

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "route": self.route,
            "summary": self.summary,
            "attempts": self.attempts,
            "widened_search": self.widened,
            "result": self.result.to_dict(),
            "trace": self.trace,
            "search": self.search,
        }


class _WithFeedback:
    """Wraps an LLM so the problems found in the last attempt are appended to its prompt."""

    def __init__(self, llm: Any, feedback: str) -> None:
        self.llm, self.feedback = llm, feedback

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.llm.generate(prompt + "\n\n" + self.feedback, **kwargs)


def _search_info(state: "AgentState") -> dict:
    """What the agent searched for and read, kept so the Excel "Search log" sheet can show it."""
    expansion = state["expansion"]
    return {
        "terms": list(expansion.terms), "terms_source": expansion.source, "note": expansion.llm_error,
        "paragraphs": [{"chunk_id": e.chunk_id, "source_file": e.source_file, "page_number": e.page_number,
                        "section": getattr(e, "section", None), "similarity": round(e.similarity, 3),
                        "matched_terms": list(e.matched_terms), "text": e.text} for e in state.get("evidence", [])],
        "paragraphs_sent": getattr(state.get("extraction"), "paragraphs_sent", None),
    }


def _feedback_from(verified: VerifiedResult) -> str:
    problems: list[str] = []
    if verified.error:
        problems.append(verified.error)
    for number, vf in enumerate(verified.findings, 1):
        if not vf.grounded:
            problems.append(f"finding #{number}: " + "; ".join(vf.check.warnings))
    problems.extend(verified.rejected)
    return (
        "IMPORTANT: your previous answer had these problems. Answer again. Fix them, or leave out any "
        "finding you cannot support with a word-for-word quote that contains the value:\n"
        + "\n".join(f"- {p}" for p in problems)
    )


def _summary(verified: VerifiedResult) -> str:
    if verified.status == STATUS_ERROR:
        return f"Could not finish: {verified.error}"
    if verified.status == STATUS_NOT_FOUND:
        return "No boundary or date cutoff was found for this field in the paragraphs read."
    lines = []
    for number, vf in enumerate(verified.findings, 1):
        f = vf.finding
        if f.kind == KIND_NUMERIC:
            what = f"{f.bound_type} {f.value} {f.unit or ''}".rstrip()
        else:
            what = f"{f.date_direction} {f.date}" + (" (month only)" if getattr(f, "date_precision", None) == "month" else "")
        page = f"p.{f.page_number}" if f.page_number else "no pagination"
        lines.append(f"#{number} {what} [{f.classification}] {'grounded' if vf.grounded else 'NOT grounded'} - {f.source_file}, {page}")
    return "\n".join(lines)


def _route(verified: VerifiedResult) -> str:
    if verified.status == STATUS_ERROR:
        return ROUTE_ERROR
    if verified.status == STATUS_NOT_FOUND:
        return ROUTE_NOT_FOUND
    return ROUTE_VERIFIED if verified.verified and not verified.needs_review else ROUTE_NEEDS_REVIEW


def build_graph(searcher: Any, llm: Any = None, config: AgentConfig | None = None, reranker: Any = None):
    """Compile the graph. `searcher` is anything with .search() (a VectorStore or HybridSearcher).
    `reranker` is optional and needs the precision-drop version of retrieve_evidence."""
    cfg = config or AgentConfig()
    if llm is None:
        from lending_agent.llm.ollama_client import OllamaClient

        llm = OllamaClient()

    def traced(name: str, fn):
        def wrapper(state: AgentState) -> dict:
            began, at = time.monotonic(), datetime.now().astimezone().isoformat(timespec="seconds")
            update = fn(state)
            entry = {"node": name, "at": at, "seconds": round(time.monotonic() - began, 3)}
            return {**update, "trace": [*state.get("trace", []), entry]}

        return wrapper

    def expand(state: AgentState) -> dict:
        return {"expansion": expand_field_terms(state["query"], llm=llm, use_llm=cfg.use_llm_expansion),
                "top_k": cfg.top_k, "attempt": 0, "widened": False, "feedback": ""}

    def retrieve(state: AgentState) -> dict:
        expansion = state["expansion"]
        extra: dict[str, Any] = {}
        if reranker is not None:
            query = state["query"].strip()
            extra = {"reranker": reranker, "rerank_query": query if " " in query else
                     f"What is the maximum, minimum or cutoff date for the {expansion.field.replace('_', ' ')}?"}
        evidence = retrieve_evidence(expansion.terms, searcher, n_per_term=state["top_k"], top_k=state["top_k"],
                                     source_file=cfg.source_file, **extra)
        return {"evidence": evidence}

    def extract(state: AgentState) -> dict:
        feedback = state.get("feedback", "")
        result = extract_conditions(
            state["expansion"].field, state["evidence"], llm=_WithFeedback(llm, feedback) if feedback else llm,
            kinds=cfg.kinds, num_ctx=cfg.num_ctx, focus=cfg.focus,
        )
        return {"extraction": result, "attempt": state["attempt"] + 1}

    def verify(state: AgentState) -> dict:
        return {"verified": verify_extraction(state["extraction"], state["evidence"])}

    def decide(state: AgentState) -> str:
        verified, attempts = state["verified"], state["attempt"]
        if verified.status == STATUS_ERROR:
            retryable = any(marker in (verified.error or "") for marker in _RETRYABLE_ERRORS)
            return "retry" if retryable and attempts < cfg.max_attempts else "respond"
        if verified.status == STATUS_NOT_FOUND:
            if not state["widened"] and cfg.widen_factor > 1 and attempts < cfg.max_attempts:
                return "widen"
            return "respond"
        if any(not vf.grounded for vf in verified.findings) and attempts < cfg.max_attempts:
            return "retry"
        return "respond"

    def prepare_retry(state: AgentState) -> dict:
        return {"feedback": _feedback_from(state["verified"])}

    def prepare_widen(state: AgentState) -> dict:
        return {"top_k": state["top_k"] * cfg.widen_factor, "widened": True, "feedback": ""}

    def respond(state: AgentState) -> dict:
        verified = state["verified"]
        answer = AgentAnswer(
            field=verified.field, route=_route(verified), summary=_summary(verified), result=verified,
            attempts=state["attempt"], widened=state["widened"], trace=list(state.get("trace", [])),
            search=_search_info(state),
        )
        return {"answer": answer}

    graph = StateGraph(AgentState)
    for name, fn in [("expand", expand), ("retrieve", retrieve), ("extract", extract), ("verify", verify),
                     ("prepare_retry", prepare_retry), ("prepare_widen", prepare_widen), ("respond", respond)]:
        graph.add_node(name, traced(name, fn))
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "verify")
    graph.add_conditional_edges("verify", decide, {"retry": "prepare_retry", "widen": "prepare_widen", "respond": "respond"})
    graph.add_edge("prepare_retry", "extract")
    graph.add_edge("prepare_widen", "retrieve")
    graph.add_edge("respond", END)
    return graph.compile()


def ask(graph: Any, query: str) -> AgentAnswer:
    """Run the compiled graph for one field name or question."""
    final = graph.invoke({"query": query, "trace": []})
    answer = final["answer"]
    answer.trace = list(final["trace"])  # the finished trace, including the respond step itself
    return answer
