import pytest

from common.config import LLMProvider, SvcSettings


@pytest.fixture
def openai_settings() -> SvcSettings:
    """Settings that force the hosted OpenAI provider."""
    return SvcSettings(llm_provider=LLMProvider.OPENAI)


@pytest.fixture
def local_llm_settings() -> SvcSettings:
    """Settings that force the configured local provider."""
    settings = SvcSettings(llm_provider=LLMProvider.LOCAL)
    if settings.llm_base_url is None:
        pytest.skip("Set LLM_BASE_URL before running the local LLM live test.")
    return settings
