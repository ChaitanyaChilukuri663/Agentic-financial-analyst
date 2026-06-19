"""Unit tests for the multi-provider LLM client (fully offline / mocked)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ledgerlens.config import Provider, Settings
from ledgerlens.llm import LLMClient, LLMError


class _Answer(BaseModel):
    """A toy structured response."""

    value: int = Field(description="The integer answer.")
    label: str = Field(description="A short label for the answer.")


def _completion(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    call = SimpleNamespace(function=function)
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.result


class _FakeEmbeddings:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.result


def _offline_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, use_os_truststore=False, **overrides)


def test_chat_structured_forces_tool_and_parses_model() -> None:
    completions = _FakeCompletions(_completion("emit", {"value": 42, "label": "ok"}))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = LLMClient(settings=_offline_settings(), clients={Provider.AZURE: fake})

    result = client.chat_structured(
        messages=[{"role": "user", "content": "compute it"}],
        response_model=_Answer,
    )

    assert isinstance(result, _Answer)
    assert (result.value, result.label) == (42, "ok")
    sent = completions.calls[0]
    assert sent["tool_choice"]["function"]["name"] == "emit"
    params = sent["tools"][0]["function"]["parameters"]
    assert {"value", "label"} <= params["properties"].keys()


def test_embed_returns_vectors() -> None:
    result = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
    )
    fake = SimpleNamespace(embeddings=_FakeEmbeddings(result))
    client = LLMClient(settings=_offline_settings(), clients={Provider.AZURE: fake})

    assert client.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_empty_input_short_circuits() -> None:
    client = LLMClient(settings=_offline_settings(), clients={Provider.AZURE: SimpleNamespace()})
    assert client.embed([]) == []


def test_chat_structured_raises_llmerror_when_all_providers_fail() -> None:
    class _Boom:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            raise ValueError("boom")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=_Boom()))
    client = LLMClient(
        settings=_offline_settings(fallback_providers=[]),
        clients={Provider.AZURE: fake},
    )

    with pytest.raises(LLMError):
        client.chat_structured(
            messages=[{"role": "user", "content": "x"}],
            response_model=_Answer,
        )
