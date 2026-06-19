"""Runtime configuration for LedgerLens, loaded from the environment / a ``.env`` file.

All settings are read from ``LEDGERLENS_``-prefixed environment variables (see
``.env.example``). No secrets are hard-coded; this module only declares shape and
defaults.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(StrEnum):
    """Supported LLM providers (all OpenAI-compatible at the wire level)."""

    AZURE = "azure"
    GITHUB = "github"
    GROQ = "groq"


class Settings(BaseSettings):
    """LedgerLens configuration resolved from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="LEDGERLENS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Provider = Field(
        default=Provider.AZURE,
        description="Primary LLM provider used for chat and embeddings.",
    )
    fallback_providers: list[Provider] = Field(
        default_factory=lambda: [Provider.GITHUB, Provider.GROQ],
        description="Providers tried in order if the primary provider fails.",
    )

    # --- Azure OpenAI (primary) ---
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI resource endpoint, e.g. https://<name>.openai.azure.com.",
    )
    azure_openai_api_key: str = Field(
        default="",
        description="Azure OpenAI API key.",
    )
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI REST API version.",
    )
    azure_chat_deployment: str = Field(
        default="gpt-4o-mini",
        description="Azure deployment name backing chat completions.",
    )
    azure_embed_deployment: str = Field(
        default="text-embedding-3-small",
        description="Azure deployment name backing embeddings.",
    )

    # --- GitHub Models (fallback) ---
    github_token: str = Field(
        default="",
        description="GitHub Models personal access token.",
    )
    github_chat_model: str = Field(
        default="gpt-4o-mini",
        description="GitHub Models chat model id.",
    )
    github_embed_model: str = Field(
        default="text-embedding-3-small",
        description="GitHub Models embedding model id.",
    )

    # --- Groq (fallback; chat only) ---
    groq_api_key: str = Field(
        default="",
        description="Groq API key.",
    )
    groq_chat_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq chat model id.",
    )

    # --- Transport ---
    use_os_truststore: bool = Field(
        default=True,
        description="Trust the OS certificate store (required behind TLS-inspecting proxies).",
    )
    request_timeout_s: float = Field(
        default=60.0,
        description="Per-request timeout in seconds.",
    )
    max_retries: int = Field(
        default=2,
        description="SDK-level retries per provider before failing over.",
    )


def get_settings() -> Settings:
    """Return a freshly-loaded :class:`Settings` instance."""
    return Settings()
