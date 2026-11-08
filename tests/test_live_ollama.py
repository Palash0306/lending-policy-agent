"""
LIVE tests: they need a running Ollama with the configured model, the real embedding model, and the sample
handbook (pub-ch-retail-lending.pdf in the project folder). They are skipped by default, and skip with a clear
message when something is missing. Run them yourself with:

    pytest -m live -v

They are slow (minutes on a laptop CPU) and check what no fake can: that the real model reads the real document.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.live

HANDBOOK = Path(__file__).resolve().parents[1] / "pub-ch-retail-lending.pdf"


@pytest.fixture(scope="module")
def llm():
    try:
        from lending_agent.llm.ollama_client import OllamaClient
    except ImportError as exc:
        pytest.skip(f"The 'ollama' Python package is not installed ({exc}). Run: pip install -r requirements.txt")
    client = OllamaClient()
    try:
        client.list_models()
        available = client.is_model_available()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable ({exc}). Start it with 'ollama serve', then run: pytest -m live")
    if not available:
        pytest.skip(f"The model '{client.settings.model}' is not installed. Run: ollama pull {client.settings.model}")
    return client


def test_real_synonym_expansion_returns_usable_terms(llm):
    from lending_agent.agent.field_expansion import SEED_SYNONYMS, expand_field_terms

    result = expand_field_terms("loan_tenure", llm=llm)
    assert result.llm_error is None, f"the model's answer was not usable: {result.llm_error}"
    assert result.source == "seed+llm"
    assert len(result.terms) > len(SEED_SYNONYMS["loan_tenure"])                      # the model added wording of its own
    assert all(len(t) <= 60 and not any(ch.isdigit() for ch in t) for t in result.terms)   # no invented numbers, short phrases


def test_real_model_finds_both_page_23_limits_as_examples_and_they_are_grounded(llm, tmp_path):
    pytest.importorskip("langgraph")
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers", reason="the real embedding model is needed (first run downloads it once)")
    if not HANDBOOK.exists():
        pytest.skip(f"{HANDBOOK.name} is not in the project folder")
    from lending_agent.agent.graph import AgentConfig, ask, build_graph
    from lending_agent.config import ChromaSettings
    from lending_agent.indexing.indexer import index_documents
    from lending_agent.indexing.vector_store import VectorStore

    store = VectorStore(settings=ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name="live_test"))
    assert index_documents([str(HANDBOOK)], store=store)[str(HANDBOOK)] > 100
    answer = ask(build_graph(store, llm=llm, config=AgentConfig()), "loan_tenure")

    found = {(f.finding.value, f.finding.classification): f for f in answer.result.findings if f.finding.page_number == 23}
    assert (72, "example") in found and (60, "example") in found, f"page 23 findings were: {sorted(found)}"
    assert found[(72, "example")].grounded and found[(60, "example")].grounded
    assert answer.route in ("verified", "needs_review")
