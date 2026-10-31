"""
Phase 4 -- Node 1 of the pipeline: expand_field_terms.

Problem: the analyst's system calls a field `loan_amount`, but the policy
document might say "sanctioned amount", "mortgage amount" or "principal".
If we only searched for the exact field name we'd miss those paragraphs.

Solution: build a list of likely phrasings for the field, then (in
retrieval.py) search for each of them. The list comes from two sources:

  1. A small built-in word list of common lending terms. Always available,
     works offline, gives the same answer every time.
  2. The local LLM (Ollama), which suggests extra phrasings -- including for
     fields the built-in list doesn't know at all. If Ollama is down or
     answers with garbage, we fall back to source 1 and report why -- the
     agent should degrade, not crash.

Nothing here ever sees document text; it only handles field names and the
analyst's question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from lending_agent.agent.llm_json import parse_llm_json


class _TextGenerator(Protocol):
    """Anything with a generate() like OllamaClient's. Lets tests pass a fake."""

    def generate(self, prompt: str, **kwargs) -> str: ...


# Common lending-policy phrasings, keyed by the internal field name.
# Deliberately free of numbers: these are only *words to search for*.
SEED_SYNONYMS: dict[str, list[str]] = {
    "loan_amount": [
        "loan amount", "sanctioned amount", "mortgage amount", "principal amount",
        "sanctioned limit", "credit amount", "facility amount", "amount of the loan",
    ],
    "loan_tenure": [
        "loan tenure", "repayment period", "loan term", "amortization period",
        "term of the loan", "maturity", "tenor", "repayment term",
    ],
    "applicant_age": [
        "applicant age", "age of the borrower", "age at application",
        "age at maturity", "borrower age", "age limit",
    ],
    "interest_rate": [
        "interest rate", "rate of interest", "lending rate", "annual percentage rate",
        "pricing", "imputed interest rate", "calculatory interest rate",
    ],
    "ltv_ratio": [
        "loan-to-value ratio", "loan to value", "LTV", "advance rate",
        "financing ratio", "lending value", "maximum advance",
    ],
    "debt_to_income": [
        "debt-to-income ratio", "debt to income", "DTI", "affordability",
        "income ratio", "burden ratio", "payment-to-income",
    ],
    "credit_score": [
        "credit score", "credit rating", "bureau score", "creditworthiness",
        "credit assessment", "risk rating",
    ],
    "down_payment": [
        "down payment", "own funds", "equity contribution", "borrower contribution",
        "minimum deposit", "own contribution",
    ],
    "income": [
        "income", "gross income", "net income", "annual income",
        "regular income", "verifiable income",
    ],
}

_FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class FieldExpansion:
    field: str                  # canonical snake_case field name, e.g. "loan_tenure"
    terms: list[str]            # phrasings to search for, best first
    source: str                 # "seed+llm", "seed", "llm" or "input"
    llm_error: str | None = None  # set when the LLM was tried but couldn't help


def _slugify(text: str, max_words: int = 4) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words[:max_words]) or "unknown_field"


def _looks_like_field_name(text: str) -> bool:
    return bool(_FIELD_NAME_RE.match(text.strip()))


def _seed_field_in_text(text: str) -> str | None:
    """Find a known field mentioned (by any of its phrasings) in a natural-language question."""
    lowered = text.lower()
    for field, terms in SEED_SYNONYMS.items():
        phrases = [field.replace("_", " ")] + [t.lower() for t in terms]
        if any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in phrases):
            return field
    return None


def _build_prompt(query: str) -> str:
    return (
        "You are helping a data assurance analyst search a bank's lending policy "
        "documents. The analyst's system uses an internal field name, but the "
        "policy document may word the same idea differently.\n\n"
        f'Input: "{query}"\n\n'
        "Reply with JSON only, in exactly this shape:\n"
        '{"field": "<snake_case internal field name>", "terms": ["<phrase>", "..."]}\n\n'
        "Rules:\n"
        '- "terms" must hold 5 to 10 short phrases (1 to 4 words each) that a '
        "bank lending policy might use for the same concept: synonyms, formal "
        "wording, and related wording.\n"
        "- Do not include numbers or specific values.\n"
        "- Do not add any explanation outside the JSON."
    )


def _clean_terms(raw_terms: object) -> list[str]:
    """Keep only short, number-free text phrases. LLM output is untrusted input."""
    if not isinstance(raw_terms, list):
        return []
    cleaned: list[str] = []
    for item in raw_terms:
        if not isinstance(item, str):
            continue
        term = re.sub(r"\s+", " ", item).strip().strip(".,;:\"'")
        if not term or len(term) > 60 or len(term.split()) > 6:
            continue
        if any(ch.isdigit() for ch in term):
            continue  # numbers don't belong in a search phrase; they could be invented values
        cleaned.append(term)
    return cleaned


def _dedupe(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def expand_field_terms(
    query: str,
    llm: _TextGenerator | None = None,
    use_llm: bool = True,
    max_terms: int = 12,
) -> FieldExpansion:
    """
    Turn a field name ("loan_tenure") or a short question ("What is the
    maximum loan tenure allowed?") into a list of phrasings to search for.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    is_field_name = _looks_like_field_name(query)
    field = query.lower() if is_field_name else _seed_field_in_text(query)
    original = query.replace("_", " ") if is_field_name else query

    seed_terms = list(SEED_SYNONYMS.get(field, [])) if field else []

    llm_terms: list[str] = []
    llm_field: str | None = None
    llm_error: str | None = None
    # The LLM is always asked (unless disabled): even for fields on the built-in list it
    # can add wording the list doesn't have. Its answer is merged after the built-in terms.
    if use_llm:
        try:
            if llm is None:
                from lending_agent.llm.ollama_client import OllamaClient

                llm = OllamaClient()
            raw = llm.generate(_build_prompt(query), format="json", temperature=0)
            data = parse_llm_json(raw)
            if data is None:
                llm_error = "The model's reply was not valid JSON, so it was ignored."
            else:
                llm_terms = _clean_terms(data.get("terms"))
                candidate = data.get("field")
                if isinstance(candidate, str) and _looks_like_field_name(candidate):
                    llm_field = candidate.lower()
                if not llm_terms:
                    llm_error = "The model returned no usable phrasings."
        except Exception as exc:  # graceful degradation is the point: any LLM failure -> fall back
            llm_error = f"Could not use the LLM: {exc}"

    final_field = field or llm_field or _slugify(query)
    terms = _dedupe([original, *seed_terms, *llm_terms])[:max_terms]

    if seed_terms and llm_terms:
        source = "seed+llm"
    elif seed_terms:
        source = "seed"
    elif llm_terms:
        source = "llm"
    else:
        source = "input"

    return FieldExpansion(field=final_field, terms=terms, source=source, llm_error=llm_error)
