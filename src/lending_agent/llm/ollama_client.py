"""
Thin wrapper around the `ollama` Python client.

Why this exists rather than calling `ollama` directly everywhere: every
later node in the pipeline (field synonym expansion in Phase 4, boundary /
date-cutoff extraction in Phase 5) will need local inference. Centralizing
the client here means connection errors, timeouts, and "model not pulled
yet" handling only need to be solved once — and it's easy to mock in tests
without a live Ollama server (see tests/test_ollama_client.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import ollama
from ollama import ResponseError

from lending_agent.config import OllamaSettings, get_settings

logger = logging.getLogger(__name__)


class OllamaConnectionError(RuntimeError):
    """The local Ollama server could not be reached at all."""


class OllamaModelNotAvailableError(RuntimeError):
    """The target model is not pulled, and auto-pull is disabled or failed."""


@dataclass
class HealthcheckResult:
    reachable: bool
    model_available: bool
    installed_models: list[str] = field(default_factory=list)
    sample_response: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.reachable and self.model_available and self.sample_response is not None


class OllamaClient:
    """Wraps the `ollama` package with the error handling and healthcheck this project needs."""

    def __init__(self, settings: OllamaSettings | None = None):
        self.settings = settings or get_settings().ollama
        self._client = ollama.Client(host=self.settings.host, timeout=self.settings.request_timeout)

    # ---- connectivity ----------------------------------------------------

    def list_models(self) -> list[str]:
        """Tags of every model currently pulled on this Ollama server."""
        try:
            response = self._client.list()
        except Exception as exc:  # connection refused, DNS failure, timeout, etc.
            raise OllamaConnectionError(f"Could not reach Ollama at {self.settings.host}: {exc}") from exc

        models = response.get("models", []) if isinstance(response, dict) else response.models
        tags = []
        for m in models:
            tag = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if tag:
                tags.append(tag)
        return tags

    def is_model_available(self, model: str | None = None) -> bool:
        model = model or self.settings.model
        installed = self.list_models()
        # Ollama tags look like "qwen2.5:7b"; also accept a bare match on the
        # name before the colon so a request for "qwen2.5:7b" is satisfied by
        # any locally installed "qwen2.5:*" variant.
        base = model.split(":")[0]
        return any(tag == model or tag.split(":")[0] == base for tag in installed)

    def ensure_model_pulled(self, model: str | None = None, progress: bool = True) -> None:
        """Pull `model` if it isn't already present, streaming progress to the logger."""
        model = model or self.settings.model
        if self.is_model_available(model):
            logger.info("Model '%s' already pulled.", model)
            return

        logger.info("Model '%s' not found locally. Pulling (this can take a while)...", model)
        try:
            stream = self._client.pull(model=model, stream=True)
            last_status = None
            for chunk in stream:
                status = chunk.get("status") if isinstance(chunk, dict) else getattr(chunk, "status", None)
                if progress and status and status != last_status:
                    logger.info("  ollama pull '%s': %s", model, status)
                    last_status = status
        except ResponseError as exc:
            raise OllamaModelNotAvailableError(f"Failed to pull model '{model}': {exc}") from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.settings.host} while pulling '{model}': {exc}"
            ) from exc

        if not self.is_model_available(model):
            raise OllamaModelNotAvailableError(
                f"Pull for '{model}' completed but the model still isn't listed by Ollama."
            )

    # ---- inference ---------------------------------------------------------

    def generate(self, prompt: str, model: str | None = None, format: str | None = None, **options) -> str:
        """Single-prompt generation. Pass format="json" to ask Ollama to constrain the output to valid JSON."""
        model = model or self.settings.model
        try:
            response = self._client.generate(
                model=model, prompt=prompt, options=options or None, format=format
            )
        except Exception as exc:
            raise OllamaConnectionError(f"Generation failed against '{model}': {exc}") from exc
        return response.get("response") if isinstance(response, dict) else response.response

    def chat(self, messages: Iterable[dict], model: str | None = None, **options) -> str:
        model = model or self.settings.model
        try:
            response = self._client.chat(model=model, messages=list(messages), options=options or None)
        except Exception as exc:
            raise OllamaConnectionError(f"Chat failed against '{model}': {exc}") from exc
        message = response.get("message") if isinstance(response, dict) else response.message
        content = message.get("content") if isinstance(message, dict) else message.content
        return content

    # ---- the Phase 1 deliverable -------------------------------------------

    def healthcheck(self, auto_pull: bool = True) -> HealthcheckResult:
        """
        Confirms, in order: (1) the Ollama server is reachable, (2) the
        configured model is available — pulling it if `auto_pull=True` and
        it's missing, and (3) a real local inference call actually succeeds.
        This is what scripts/verify_ollama.py calls.
        """
        try:
            installed = self.list_models()
        except OllamaConnectionError as exc:
            return HealthcheckResult(reachable=False, model_available=False, error=str(exc))

        model_available = self.is_model_available()
        if not model_available and auto_pull:
            try:
                self.ensure_model_pulled()
                model_available = True
                installed = self.list_models()
            except (OllamaConnectionError, OllamaModelNotAvailableError) as exc:
                return HealthcheckResult(
                    reachable=True, model_available=False, installed_models=installed, error=str(exc)
                )

        if not model_available:
            return HealthcheckResult(
                reachable=True,
                model_available=False,
                installed_models=installed,
                error=f"Model '{self.settings.model}' is not pulled and auto_pull=False.",
            )

        try:
            sample = self.generate("Reply with exactly one word: OK", temperature=0)
        except OllamaConnectionError as exc:
            return HealthcheckResult(
                reachable=True, model_available=True, installed_models=installed, error=str(exc)
            )

        return HealthcheckResult(
            reachable=True,
            model_available=True,
            installed_models=installed,
            sample_response=sample.strip(),
        )
