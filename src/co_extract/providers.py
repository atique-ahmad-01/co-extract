"""Choose the LLM provider: LLM_PROVIDER, or whichever API key is set."""

from __future__ import annotations

import os
from pathlib import Path

PROVIDERS = ("openai", "anthropic")


def available_providers() -> list[str]:
    found = []
    if os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        found.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        found.append("anthropic")
    return found


def default_provider() -> str | None:
    chosen = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if chosen in PROVIDERS:
        return chosen
    found = available_providers()
    return found[0] if found else None


def make_client(provider: str | None = None, model: str | None = None, cache: bool = True,
                cache_dir: str | Path | None = None, effort: str | None = None):
    provider = provider or default_provider()
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        client = AnthropicClient(model=model, effort=effort)
    elif provider == "openai":
        from .llm import OpenAICompatibleClient
        client = OpenAICompatibleClient(model=model)
    else:
        raise RuntimeError("no LLM configured: set LLM_API_KEY or ANTHROPIC_API_KEY in .env")
    client.provider = provider
    if cache:
        from .llm import CachedClient
        client = CachedClient(client, cache_dir or os.environ.get("LLM_CACHE_DIR", ".cache/llm"))
    return client
