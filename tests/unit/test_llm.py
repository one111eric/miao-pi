import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from common.config import LLMProvider, SvcSettings
from common.llm import LLM

LLM_ENVIRONMENT_VARIABLES = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
)


class UnitTestSettings(SvcSettings):
    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture(autouse=True)
def clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from developer and CI environments."""
    for variable in LLM_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_llm_is_a_langchain_chat_model() -> None:
    llm = LLM(
        "local-model",
        base_url="http://localhost:11434/v1",
        settings=UnitTestSettings(
            llm_api_key=SecretStr("local-test-key"),
        ),
    )

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "local-model"
    assert llm.openai_api_base == "http://localhost:11434/v1"
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "local-test-key"


def test_llm_forwards_langchain_options() -> None:
    llm = LLM(
        "gpt-test",
        api_key="test-key",
        settings=UnitTestSettings(),
        temperature=0.2,
        max_tokens=25,
    )

    assert llm.temperature == 0.2
    assert llm.max_tokens == 25
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "test-key"


def test_llm_uses_the_selected_provider_settings() -> None:
    settings = UnitTestSettings(
        llm_provider=LLMProvider.LOCAL,
        llm_model="qwen2.5",
        llm_base_url="http://localhost:11434/v1",
    )

    llm = LLM(settings=settings)

    assert llm.model_name == "qwen2.5"
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_rejects_an_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        LLM("", settings=UnitTestSettings())


def test_local_provider_requires_a_base_url() -> None:
    settings = UnitTestSettings(llm_provider=LLMProvider.LOCAL)

    with pytest.raises(ValueError, match="llm_base_url"):
        LLM(settings=settings)
