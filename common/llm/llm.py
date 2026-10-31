"""LangChain chat model configured for OpenAI-compatible APIs."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from common.config import LLMProvider, SvcSettings

type APIKey = SecretStr | Callable[[], str] | Callable[[], Awaitable[str]]


class LLM(ChatOpenAI):
    """A ``ChatOpenAI`` usable with OpenAI or a compatible local server.

    This is directly usable as a LangChain runnable or as a LangGraph node,
    exposing the standard ``invoke``, ``ainvoke``, and ``stream`` APIs.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | APIKey | None = None,
        base_url: str | None = None,
        settings: SvcSettings | None = None,
        **kwargs: Any,
    ) -> None:
        settings = settings or SvcSettings()
        if model is None:
            model = settings.llm_model
        if base_url is None and settings.llm_provider is LLMProvider.LOCAL:
            base_url = settings.llm_base_url
        if api_key is None:
            api_key = (
                settings.llm_api_key
                if base_url is not None
                else settings.openai_api_key
            )

        if not model:
            raise ValueError("model must not be empty")
        if settings.llm_provider is LLMProvider.LOCAL and base_url is None:
            raise ValueError("llm_base_url must be set when llm_provider is local")

        # OpenAI discovers OPENAI_API_KEY when this remains None. Most local
        # compatible servers ignore the key, while the OpenAI client requires one.
        if base_url is not None and api_key is None:
            api_key = SecretStr("local-llm")
        elif isinstance(api_key, str):
            api_key = SecretStr(api_key)

        # Explicit arguments take precedence over the configured defaults.
        kwargs.setdefault("timeout", settings.llm_timeout)
        kwargs.setdefault("max_retries", settings.llm_max_retries)

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
