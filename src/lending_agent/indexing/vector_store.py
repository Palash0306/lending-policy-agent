"""
Thin wrapper around a local, persistent Chroma collection. Chunks go in,
similarity search results come out -- this is the retrieval half of RAG
(embedding + storage here in Phase 3; field synonym expansion and the
extraction prompt that call this live in Phase 4/5).
"""
from __future__ import annotations

from dataclasses import dataclass

from lending_agent.config import ChromaSettings, get_settings
from lending_agent.indexing.embeddings import EmbeddingModel
from lending_agent.ingestion.models import Chunk


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source_file: str
    page_number: int | None
    doc_type: str
    distance: float  # lower = more similar (cosine distance, since the collection is configured for cosine space)

    @property
    def similarity(self) -> float:
        """A rough 0-1 similarity score for display purposes -- not a calibrated probability."""
        return max(0.0, 1.0 - self.distance)


class VectorStore:
    """
    Wraps a Chroma PersistentClient collection. The embedding model is
    injected (defaults to a real EmbeddingModel) so tests can swap in a
    lightweight fake without needing network access or a model download.
    """

    def __init__(self, settings: ChromaSettings | None = None, embedding_model: EmbeddingModel | None = None):
        self.settings = settings or get_settings().chroma
        self.embedding_model = embedding_model or EmbeddingModel()
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.settings.persist_directory)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert chunks into the collection. Upsert means re-indexing the same
        file is safe to re-run -- chunks with the same chunk_id simply get overwritten
        rather than duplicated. Returns the number of chunks added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "doc_type": c.doc_type,
                "page_number": c.page_number or 0,  # Chroma metadata can't store None
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        collection = self._get_collection()
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(chunks)

    def count(self) -> int:
        return self._get_collection().count()

    def search(self, query: str, n_results: int = 5, source_file: str | None = None) -> list[SearchResult]:
        """Semantic search: embed the query, return the n_results closest chunks.
        Optionally restrict the search to one source_file."""
        query_embedding = self.embedding_model.embed_one(query)
        where = {"source_file": source_file} if source_file else None

        collection = self._get_collection()
        raw = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)

        results: list[SearchResult] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            page = meta.get("page_number") or None  # 0 was standing in for "no pagination"
            results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc,
                    source_file=meta.get("source_file", ""),
                    page_number=page,
                    doc_type=meta.get("doc_type", ""),
                    distance=dist,
                )
            )
        return results

    def sources(self) -> dict[str, int]:
        """Every indexed file and how many chunks it has: {source_file: chunk_count}."""
        collection = self._get_collection()
        counts: dict[str, int] = {}
        offset, page = 0, 1000
        while True:
            raw = collection.get(include=["metadatas"], limit=page, offset=offset)
            ids = raw.get("ids") or []
            if not ids:
                break
            for meta in raw.get("metadatas") or []:
                name = (meta or {}).get("source_file", "")
                counts[name] = counts.get(name, 0) + 1
            offset += len(ids)
        return dict(sorted(counts.items()))

    def delete_source(self, source_file: str) -> int:
        """Remove every chunk of one file. Returns how many chunks were removed (0 if it was not indexed)."""
        collection = self._get_collection()
        ids = collection.get(where={"source_file": source_file}, include=["metadatas"]).get("ids") or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """Delete the collection entirely. Used by tests and the CLI's --reset flag."""
        client = self._get_client()
        try:
            client.delete_collection(self.settings.collection_name)
        except Exception:
            pass  # collection may not exist yet -- fine, nothing to delete
        self._collection = None
