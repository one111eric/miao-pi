"""Opt-in smoke test for the configured OpenAI LLM."""

import os

import pytest
from langchain_core.messages import AIMessage

from common.config import SvcSettings
from common.llm import LLM


@pytest.mark.openai_live
@pytest.mark.skipif(
    os.environ.get("RUN_OPENAI_LIVE_TESTS") != "1",
    reason="Set RUN_OPENAI_LIVE_TESTS=1 to make a paid OpenAI API request.",
)
def test_configured_llm_can_call_openai(openai_settings: SvcSettings) -> None:
    response = LLM(settings=openai_settings, temperature=0, max_tokens=5).invoke(
        "Reply with the single word: ok"
    )
    print(response)

    assert isinstance(response, AIMessage)
    assert response.content


@pytest.mark.local_llm_live
@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_LLM_LIVE_TESTS") != "1",
    reason="Set RUN_LOCAL_LLM_LIVE_TESTS=1 to call the configured local model.",
)
def test_configured_llm_can_call_local_model(local_llm_settings: SvcSettings) -> None:
    response = LLM(settings=local_llm_settings, temperature=0, max_tokens=5).invoke(
        "Reply with the single word: ok"
    )

    assert isinstance(response, AIMessage)
    assert response.content
