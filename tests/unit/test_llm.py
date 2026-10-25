import pytest
from langchain_openai import ChatOpenAI

from common.config import LLMProvider, SvcSettings
from common.llm import LLM


def test_llm_is_a_langchain_chat_model() -> None:
    llm = LLM(
        "local-model",
        base_url="http://localhost:11434/v1",
        settings=SvcSettings(llm_api_key="local-test-key"),
    )

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "local-model"
    assert llm.openai_api_base == "http://localhost:11434/v1"
    assert llm.openai_api_key.get_secret_value() == "local-test-key"


def test_llm_forwards_langchain_options() -> None:
    llm = LLM("gpt-test", api_key="test-key", temperature=0.2, max_tokens=25)

    assert llm.temperature == 0.2
    assert llm.max_tokens == 25
    assert llm.openai_api_key.get_secret_value() == "test-key"


def test_llm_uses_the_selected_provider_settings() -> None:
    settings = SvcSettings(
        llm_provider=LLMProvider.LOCAL,
        llm_model="qwen2.5",
        llm_base_url="http://localhost:11434/v1",
    )

    llm = LLM(settings=settings)

    assert llm.model_name == "qwen2.5"
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_rejects_an_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        LLM("")


def test_local_provider_requires_a_base_url() -> None:
    settings = SvcSettings(llm_provider=LLMProvider.LOCAL)

    with pytest.raises(ValueError, match="llm_base_url"):
        LLM(settings=settings)
