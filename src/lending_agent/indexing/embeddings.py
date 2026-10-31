"""
Local embedding model wrapper (sentence-transformers).

Loading a transformer model is slow (a second or more) and only needs to
happen once per process, so the model is loaded lazily on first use and
cached on the instance rather than reloaded per call. This class is also
the seam the tests use to avoid a real model download: anything that needs
embeddings takes an `EmbeddingModel`-shaped object as a constructor
argument, so tests can pass a lightweight fake instead.
"""
from __future__ import annotations

from typing import Iterable

from lending_agent.config import EmbeddingSettings, get_settings


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model can't be loaded (not installed, no network on first run, etc.)."""


class EmbeddingModel:
    """Wraps a sentence-transformers model for encoding chunk/query text into vectors."""

    def __init__(self, settings: EmbeddingSettings | None = None):
        self.settings = settings or get_settings().embedding
        self._model = None  # lazy-loaded on first embed() call

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        try:
            self._model = SentenceTransformer(self.settings.model_name, device=self.settings.device)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model '{self.settings.model_name}'. If this is the "
                f"first run, it needs internet access once to download the model weights from "
                f"Hugging Face; after that it's fully offline. Original error: {exc}"
            ) from exc

        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text, same order, [] for empty input."""
        texts = list(texts)
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._load().get_sentence_embedding_dimension()
