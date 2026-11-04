import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

# Mirrors BaseChatModel.bind_tools, whose signature uses Any.
type BindableTool = dict[str, Any] | type | Callable[..., Any] | BaseTool


class EchoChatModel(BaseChatModel):
    """Chat model that echoes the last message, so tests never call a real LLM.

    Every call's input messages are recorded in ``calls``. Tools can be bound,
    but it never calls them; their names are recorded in ``bound_tools``.
    """

    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "echo"

    def bind_tools(
        self,
        tools: Sequence[BindableTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = [tool.name for tool in tools if isinstance(tool, BaseTool)]
        return self

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


class ScriptedChatModel(BaseChatModel):
    """Chat model that returns ``responses`` in order, tool calls included.

    Every call's input messages are recorded in ``calls``. Running out of
    responses is an error, so a test can't loop forever by accident.
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[BindableTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if len(self.calls) == len(self.responses):
            raise AssertionError("ScriptedChatModel ran out of responses")
        response = self.responses[len(self.calls)].model_copy()
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=response)])


class GatedChatModel(BaseChatModel):
    """Holds every call until ``release`` is set, so a run stays in flight."""

    started: asyncio.Event = Field(default_factory=asyncio.Event)
    release: asyncio.Event = Field(default_factory=asyncio.Event)

    @property
    def _llm_type(self) -> str:
        return "gated"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("only the async path is used")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.started.set()
        await self.release.wait()
        message = AIMessage(content=f"Received: {messages[-1].text}")
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, call_id: str, **args: object) -> AIMessage:
    """A model response that asks for one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
