"""
Configuration for the Lending Policy Data Assurance Agent.

A single dataclass tree so every node in the eventual LangGraph pipeline
pulls config from one place, rather than scattering `os.getenv` calls
across the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is a convenience, not a hard requirement — env vars set
    # any other way (shell, Docker, systemd) still work without it.
    pass


@dataclass(frozen=True)
class OllamaSettings:
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300"))
    pull_timeout: float = float(os.getenv("OLLAMA_PULL_TIMEOUT", "1800"))


@dataclass(frozen=True)
class EmbeddingSettings:
    """
    Local embedding model settings (Phase 3). Uses sentence-transformers,
    not Ollama, per the project's tech stack -- this runs as an ordinary
    Python/PyTorch model in-process, fully offline once the model weights
    are cached locally (first run downloads them once; every run after
    that needs no network at all).
    """

    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")


@dataclass(frozen=True)
class ChromaSettings:
    """Local, persistent Chroma vector store settings (Phase 3)."""

    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "lending_policies")


@dataclass(frozen=True)
class Settings:
    ollama: OllamaSettings
    embedding: EmbeddingSettings
    chroma: ChromaSettings


def get_settings() -> Settings:
    """Single entry point for config. Later phases add fields here, not new globals."""
    return Settings(
        ollama=OllamaSettings(),
        embedding=EmbeddingSettings(),
        chroma=ChromaSettings(),
    )
