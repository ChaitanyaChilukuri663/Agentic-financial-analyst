"""Hand-rolled multi-provider LLM client.

Uses the official OpenAI SDK as transport for Azure OpenAI (primary) and the
OpenAI-compatible GitHub Models / Groq endpoints (fallbacks). The *abstraction* —
provider fail-over, structured-output forcing, and OS-trust-store SSL — is
hand-rolled here. It exposes two operations:

- :meth:`LLMClient.chat_structured` forces the model to call a tool whose schema
  is a Pydantic model, then validates the returned arguments into that model. The
  model never free-forms its structured output.
- :meth:`LLMClient.embed` returns embedding vectors for a batch of texts.

No business logic lives here; this is pure provider plumbing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import AzureOpenAI, OpenAI, OpenAIError
from pydantic import BaseModel

from ledgerlens.config import Provider, Settings, get_settings
from ledgerlens.net import build_http_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Providers that expose an embeddings endpoint (Groq is chat-only).
_EMBED_PROVIDERS = (Provider.AZURE, Provider.GITHUB)


class LLMError(RuntimeError):
    """Raised when every configured provider fails for a request."""


class LLMClient:
    """Multi-provider chat + embedding client with structured-output forcing."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        clients: dict[Provider, OpenAI] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Pre-built clients can be injected (used in tests to avoid network / the SDK).
        self._clients: dict[Provider, OpenAI] = dict(clients) if clients else {}

    # -- provider ordering --------------------------------------------------
    def _provider_order(self) -> list[Provider]:
        order = [self.settings.provider]
        order += [p for p in self.settings.fallback_providers if p != self.settings.provider]
        # Only try providers we have credentials for (or an injected client, used in tests).
        return [p for p in order if self._has_credentials(p)]

    def _has_credentials(self, provider: Provider) -> bool:
        if provider in self._clients:
            return True
        s = self.settings
        if provider is Provider.AZURE:
            return bool(s.azure_openai_api_key and s.azure_openai_endpoint)
        if provider is Provider.GITHUB:
            return bool(s.github_token)
        if provider is Provider.GROQ:
            return bool(s.groq_api_key)
        return False

    # -- lazy client construction ------------------------------------------
    def _client_for(self, provider: Provider) -> OpenAI:
        if provider not in self._clients:
            self._clients[provider] = self._build_client(provider)
        return self._clients[provider]

    def _build_client(self, provider: Provider) -> OpenAI:
        s = self.settings
        http_client = build_http_client(
            use_os_truststore=s.use_os_truststore,
            timeout_s=s.request_timeout_s,
        )
        if provider is Provider.AZURE:
            return AzureOpenAI(
                azure_endpoint=s.azure_openai_endpoint,
                api_key=s.azure_openai_api_key,
                api_version=s.azure_openai_api_version,
                max_retries=max(s.max_retries, 6),
                http_client=http_client,
            )
        if provider is Provider.GITHUB:
            return OpenAI(
                base_url=GITHUB_MODELS_BASE_URL,
                api_key=s.github_token,
                max_retries=max(s.max_retries, 6),
                http_client=http_client,
            )
        if provider is Provider.GROQ:
            return OpenAI(
                base_url=GROQ_BASE_URL,
                api_key=s.groq_api_key,
                max_retries=max(s.max_retries, 6),
                http_client=http_client,
            )
        raise ValueError(f"unknown provider: {provider}")

    # -- model-name resolution ---------------------------------------------
    def _chat_model(self, provider: Provider) -> str:
        return {
            Provider.AZURE: self.settings.azure_chat_deployment,
            Provider.GITHUB: self.settings.github_chat_model,
            Provider.GROQ: self.settings.groq_chat_model,
        }[provider]

    def _embed_model(self, provider: Provider) -> str:
        models = {
            Provider.AZURE: self.settings.azure_embed_deployment,
            Provider.GITHUB: self.settings.github_embed_model,
        }
        if provider not in models:
            raise ValueError(f"provider {provider.value} does not support embeddings")
        return models[provider]

    # -- public API ---------------------------------------------------------
    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        tool_name: str = "emit",
        temperature: float = 0.0,
    ) -> T:
        """Force a tool call shaped like ``response_model`` and return it validated.

        Tries the primary provider, then each fallback, raising :class:`LLMError`
        only if all of them fail.
        """
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": response_model.__doc__ or f"Return a {response_model.__name__}.",
                "parameters": response_model.model_json_schema(),
            },
        }
        last_error: Exception | None = None
        for provider in self._provider_order():
            try:
                client = self._client_for(provider)
                completion = client.chat.completions.create(
                    model=self._chat_model(provider),
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                    temperature=temperature,
                )
                return self._parse_tool_call(completion, response_model, tool_name)
            except (OpenAIError, ValueError) as exc:
                last_error = exc
                logger.warning("provider %s failed for chat_structured: %s", provider.value, exc)
        raise LLMError("all providers failed for chat_structured") from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (empty input -> empty list)."""
        if not texts:
            return []
        order = [p for p in self._provider_order() if p in _EMBED_PROVIDERS]
        last_error: Exception | None = None
        for provider in order:
            try:
                client = self._client_for(provider)
                response = client.embeddings.create(model=self._embed_model(provider), input=texts)
                return [item.embedding for item in response.data]
            except (OpenAIError, ValueError) as exc:
                last_error = exc
                logger.warning("provider %s failed for embed: %s", provider.value, exc)
        raise LLMError("all providers failed for embed") from last_error

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _parse_tool_call(completion: Any, response_model: type[T], tool_name: str) -> T:
        choice = completion.choices[0]
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            raise ValueError("model returned no tool call")
        call = tool_calls[0]
        if call.function.name != tool_name:
            raise ValueError(f"unexpected tool call: {call.function.name}")
        arguments = json.loads(call.function.arguments)
        return response_model.model_validate(arguments)
