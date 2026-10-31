"""
Phase 3 tests for the embedding model wrapper. These inject a fake
`sentence_transformers` module via sys.modules rather than requiring the
real package + a Hugging Face model download, so they run fully offline
and fast -- the same mocking pattern used for OllamaClient in Phase 1.
"""
from __future__ import annotations

import sys
import types

import pytest

from lending_agent.config import EmbeddingSettings
from lending_agent.indexing.embeddings import EmbeddingModel, EmbeddingModelError


class _FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer."""

    instances_created = 0

    def __init__(self, model_name, device=None):
        _FakeSentenceTransformer.instances_created += 1
        self.model_name = model_name
        self.device = device

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        import numpy as np

        # Deterministic fake "embedding": [character count, word count].
        return np.array([[float(len(t)), float(len(t.split()))] for t in texts])

    def get_sentence_embedding_dimension(self):
        return 2


@pytest.fixture(autouse=True)
def reset_instance_counter():
    _FakeSentenceTransformer.instances_created = 0
    yield


@pytest.fixture
def fake_st_module(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def settings():
    return EmbeddingSettings(model_name="fake-model", device="cpu")


class TestEmbed:
    def test_embed_empty_list_returns_empty_without_loading_model(self, settings, monkeypatch):
        # sentence_transformers is deliberately NOT patched to something usable
        # here -- if embed([]) tried to load a model anyway, this would raise,
        # proving the empty-input short-circuit actually works.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        model = EmbeddingModel(settings)
        assert model.embed([]) == []

    def test_embed_returns_one_vector_per_text_in_order(self, settings, fake_st_module):
        model = EmbeddingModel(settings)
        vectors = model.embed(["hello world", "a"])
        assert len(vectors) == 2
        assert vectors[0] == [11.0, 2.0]
        assert vectors[1] == [1.0, 1.0]

    def test_embed_one_returns_a_single_vector(self, settings, fake_st_module):
        model = EmbeddingModel(settings)
        assert model.embed_one("hi there") == [8.0, 2.0]

    def test_model_is_loaded_once_and_cached_across_calls(self, settings, fake_st_module):
        model = EmbeddingModel(settings)
        model.embed(["a"])
        model.embed(["b"])
        _ = model.dimension
        assert _FakeSentenceTransformer.instances_created == 1


class TestDimension:
    def test_returns_model_embedding_dimension(self, settings, fake_st_module):
        model = EmbeddingModel(settings)
        assert model.dimension == 2


class TestErrorHandling:
    def test_raises_embedding_model_error_when_package_not_installed(self, settings, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        model = EmbeddingModel(settings)
        with pytest.raises(EmbeddingModelError):
            model.embed(["some text"])

    def test_raises_embedding_model_error_when_model_load_fails(self, settings, monkeypatch):
        module = types.ModuleType("sentence_transformers")

        class _Broken:
            def __init__(self, *a, **k):
                raise OSError("couldn't connect to huggingface.co")

        module.SentenceTransformer = _Broken
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

        model = EmbeddingModel(settings)
        with pytest.raises(EmbeddingModelError):
            model.embed(["some text"])
