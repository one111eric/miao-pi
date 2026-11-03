from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class EchoChatModel(BaseChatModel):
    """Chat model that echoes the last message, so tests never call a real LLM.

    Every call's input messages are recorded in ``calls``.
    """

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        message = AIMessage(
            content=f"Received: {messages[-1].text}",
            usage_metadata={
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
