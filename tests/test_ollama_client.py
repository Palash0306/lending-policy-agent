"""
Phase 1 tests. These mock the `ollama` package entirely, so they run
without a live Ollama server -- suitable for CI. scripts/verify_ollama.py
is the tool for testing against a *real* server.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from lending_agent.config import OllamaSettings
from lending_agent.llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotAvailableError,
)


@pytest.fixture
def settings():
    return OllamaSettings(host="http://localhost:11434", model="qwen2.5:7b")


def _client_with_mock(settings):
    with patch("lending_agent.llm.ollama_client.ollama.Client") as MockClient:
        instance = MockClient.return_value
        client = OllamaClient(settings)
        return client, instance


class TestListModels:
    def test_returns_tags_from_dict_response(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:7b"}, {"model": "llama3.1:8b"}]}
        assert client.list_models() == ["qwen2.5:7b", "llama3.1:8b"]

    def test_connection_failure_raises_ollama_connection_error(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.side_effect = ConnectionError("refused")
        with pytest.raises(OllamaConnectionError):
            client.list_models()


class TestModelAvailability:
    def test_exact_match(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:7b"}]}
        assert client.is_model_available("qwen2.5:7b") is True

    def test_base_name_match_ignores_tag(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:14b"}]}
        assert client.is_model_available("qwen2.5:7b") is True

    def test_no_match(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "llama3.1:8b"}]}
        assert client.is_model_available("qwen2.5:7b") is False


class TestEnsurePulled:
    def test_skips_pull_if_already_available(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:7b"}]}
        client.ensure_model_pulled("qwen2.5:7b")
        mock.pull.assert_not_called()

    def test_pulls_if_missing_and_streams_progress(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.side_effect = [
            {"models": []},
            {"models": [{"model": "qwen2.5:7b"}]},
        ]
        mock.pull.return_value = iter([{"status": "downloading"}, {"status": "success"}])
        client.ensure_model_pulled("qwen2.5:7b")
        mock.pull.assert_called_once_with(model="qwen2.5:7b", stream=True)

    def test_raises_if_still_missing_after_pull(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": []}
        mock.pull.return_value = iter([{"status": "success"}])
        with pytest.raises(OllamaModelNotAvailableError):
            client.ensure_model_pulled("qwen2.5:7b")


class TestGenerateAndChat:
    def test_generate_returns_response_text(self, settings):
        client, mock = _client_with_mock(settings)
        mock.generate.return_value = {"response": "hello"}
        assert client.generate("hi") == "hello"

    def test_chat_returns_message_content(self, settings):
        client, mock = _client_with_mock(settings)
        mock.chat.return_value = {"message": {"content": "hi there"}}
        assert client.chat([{"role": "user", "content": "hi"}]) == "hi there"

    def test_generate_passes_json_format_through(self, settings):
        client, mock = _client_with_mock(settings)
        mock.generate.return_value = {"response": "{}"}
        client.generate("hi", format="json")
        assert mock.generate.call_args.kwargs["format"] == "json"

    def test_generate_wraps_exceptions(self, settings):
        client, mock = _client_with_mock(settings)
        mock.generate.side_effect = TimeoutError("slow")
        with pytest.raises(OllamaConnectionError):
            client.generate("hi")


class TestHealthcheck:
    def test_full_success(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:7b"}]}
        mock.generate.return_value = {"response": "OK"}
        result = client.healthcheck()
        assert result.ok
        assert result.sample_response == "OK"

    def test_server_unreachable(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.side_effect = ConnectionError("refused")
        result = client.healthcheck()
        assert not result.reachable
        assert not result.ok
        assert result.error is not None

    def test_model_missing_and_auto_pull_disabled(self, settings):
        client, mock = _client_with_mock(settings)
        mock.list.return_value = {"models": []}
        result = client.healthcheck(auto_pull=False)
        assert result.reachable
        assert not result.model_available
        assert not result.ok
