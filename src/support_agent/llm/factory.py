"""Provider factory: one config switch (LLM_PROVIDER) selects the chat model and embeddings."""

from __future__ import annotations

import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from support_agent.core.settings import OPENROUTER_BASE_URL, Settings, get_settings

_LIMITERS: dict[float, BaseRateLimiter] = {}


def rate_limiter_for(settings: Settings) -> BaseRateLimiter | None:
    """One limiter per rate, shared by every model built in this process."""
    rpm = settings.llm_requests_per_minute or settings.app.llm.requests_per_minute
    if not rpm:
        return None
    if rpm not in _LIMITERS:
        _LIMITERS[rpm] = InMemoryRateLimiter(
            requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=1
        )
    return _LIMITERS[rpm]


class MissingCredentials(RuntimeError):
    """The selected provider has no API key configured."""


def _require_key(settings: Settings, provider: str, env_name: str) -> str:
    key = settings.api_key_for(provider)
    if not key:
        raise MissingCredentials(
            f"{env_name} is not set but provider {provider!r} is selected. "
            f"Set it in .env or choose another LLM_PROVIDER."
        )
    return key


# Claude models that reject sampling parameters (HTTP 400 "`temperature` is deprecated for this
# model"): Fable and Mythos, the 5.x Opus/Sonnet/Haiku, and Opus 4.7/4.8. Older ones accept it.
_NO_SAMPLING = re.compile(r"^claude-(?:(?:fable|mythos)-|(?:opus|sonnet|haiku)-5|opus-4-[78])")


def anthropic_temperature(model: str, temperature: float) -> float | None:
    """The temperature to send, or None when the model does not take one."""
    return None if _NO_SAMPLING.match(model) else temperature


def get_chat_model(*, streaming: bool = True, settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    cfg = settings.app.llm
    provider = settings.llm_provider
    model = settings.chat_model_name
    limiter = rate_limiter_for(settings)

    if provider in ("openai", "openrouter"):
        from langchain_openai import ChatOpenAI

        env = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        base_url = OPENROUTER_BASE_URL if provider == "openrouter" else None
        return ChatOpenAI(
            model=model,
            api_key=_require_key(settings, provider, env),  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,  # type: ignore[call-arg]
            timeout=cfg.timeout_seconds,
            streaming=streaming,
            stream_usage=True,
            base_url=base_url,
            rate_limiter=limiter,
            max_retries=cfg.max_retries,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model_name=model,
            api_key=_require_key(settings, provider, "ANTHROPIC_API_KEY"),  # type: ignore[arg-type]
            temperature=anthropic_temperature(model, cfg.temperature),
            max_tokens_to_sample=cfg.max_output_tokens,
            timeout=cfg.timeout_seconds,
            streaming=streaming,
            rate_limiter=limiter,
            max_retries=cfg.max_retries,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_require_key(settings, provider, "GOOGLE_API_KEY"),
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            timeout=cfg.timeout_seconds,
            rate_limiter=limiter,
            max_retries=cfg.max_retries,
        )
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    provider = settings.embedding_provider
    model = settings.embedding_model_name

    if provider == "local":
        from support_agent.rag.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=model, cache_dir=settings.app.embeddings.cache_dir)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            api_key=_require_key(settings, "openai", "OPENAI_API_KEY"),  # type: ignore[arg-type]
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(  # type: ignore[call-arg]
            model=model, google_api_key=_require_key(settings, "gemini", "GOOGLE_API_KEY")
        )
    raise ValueError(f"Unsupported embedding provider: {provider!r}")
