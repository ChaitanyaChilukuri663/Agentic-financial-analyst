"""Tests for Settings loading (independent of any local .env)."""

from __future__ import annotations

from ledgerlens.config import Provider, Settings


def test_defaults_match_brief() -> None:
    s = Settings(_env_file=None)
    assert s.provider is Provider.AZURE
    assert s.fallback_providers == [Provider.GITHUB, Provider.GROQ]
    assert s.azure_chat_deployment == "gpt-4o-mini"
    assert s.azure_embed_deployment == "text-embedding-3-small"
    assert s.use_os_truststore is True


def test_env_prefix_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LEDGERLENS_PROVIDER", "groq")
    monkeypatch.setenv("LEDGERLENS_GROQ_API_KEY", "secret-token")
    s = Settings(_env_file=None)
    assert s.provider is Provider.GROQ
    assert s.groq_api_key == "secret-token"
