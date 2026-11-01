import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from common.config import LLMProvider, SvcSettings
from common.llm import LLM

LLM_ENVIRONMENT_VARIABLES = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "OPENAI_API_KEY",
)


def unit_settings(
    *,
    provider: LLMProvider = LLMProvider.OPENAI,
    model: str = "test-model",
    api_key: SecretStr | None = None,
    base_url: str | None = None,
) -> SvcSettings:
    """Build settings whose LLM values cannot be overridden by the environment."""
    return SvcSettings(
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_base_url=base_url,
        openai_api_key=None,
    )


@pytest.fixture(autouse=True)
def clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from developer and CI environments."""
    for variable in LLM_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_llm_is_a_langchain_chat_model() -> None:
    llm = LLM(
        "local-model",
        base_url="http://localhost:11434/v1",
        settings=unit_settings(
            api_key=SecretStr("local-test-key"),
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
        settings=unit_settings(),
        temperature=0.2,
        max_tokens=25,
    )

    assert llm.temperature == 0.2
    assert llm.max_tokens == 25
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "test-key"


def test_llm_uses_the_selected_provider_settings() -> None:
    settings = unit_settings(
        provider=LLMProvider.LOCAL,
        model="qwen2.5",
        base_url="http://localhost:11434/v1",
    )

    llm = LLM(settings=settings)

    assert llm.model_name == "qwen2.5"
    assert llm.openai_api_base == "http://localhost:11434/v1"


def test_llm_uses_the_configured_timeout_and_retries() -> None:
    settings = SvcSettings(
        llm_provider=LLMProvider.OPENAI,
        llm_timeout=12.5,
        llm_max_retries=1,
    )

    llm = LLM("gpt-test", api_key="test-key", settings=settings)

    assert llm.request_timeout == 12.5
    assert llm.max_retries == 1


def test_explicit_timeout_and_retries_override_settings() -> None:
    settings = SvcSettings(
        llm_provider=LLMProvider.OPENAI,
        llm_timeout=12.5,
        llm_max_retries=1,
    )

    llm = LLM(
        "gpt-test",
        api_key="test-key",
        settings=settings,
        timeout=5,
        max_retries=0,
    )

    assert llm.request_timeout == 5
    assert llm.max_retries == 0


@pytest.mark.parametrize(
    "overrides",
    [{"llm_timeout": 0}, {"llm_max_retries": -1}],
)
def test_rejects_invalid_timeout_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SvcSettings.model_validate(overrides)


def test_rejects_an_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        LLM("", settings=unit_settings())


def test_local_provider_requires_a_base_url() -> None:
    settings = unit_settings(provider=LLMProvider.LOCAL)

    with pytest.raises(ValueError, match="llm_base_url"):
        LLM(settings=settings)
