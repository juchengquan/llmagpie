"""Provider-agnostic LLM node abstraction.

Concrete provider nodes (OpenAI, Anthropic, Ollama, …) subclass
:class:`BaseLLMNode` and implement :meth:`BaseLLMNode._complete` to
return a normalized :class:`LLMResponse`. The base class supplies the
common ``async_call`` driver loop, including tool-calling iteration."""

from __future__ import annotations

import json
from typing import Any

from llmagpie.base.node import BaseNode, MakeNode
from pydantic import BaseModel, ConfigDict, Field


class LLMUsage(BaseModel):
    """Per-call token usage. All counts default to 0 for providers that
    don't return usage data."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Normalized response shape across LLM providers.

    Fields:
        content: The generated text content (empty string if the
            provider returned only tool calls).
        tool_calls: List of tool-call dicts in OpenAI-style shape
            ``{"id": ..., "type": "function", "function": {"name": ...,
            "arguments": ...}}``. Empty list if none.
        finish_reason: Provider-specific finish indicator (``"stop"``,
            ``"length"``, ``"tool_calls"``, …). ``None`` if unknown.
        model: Model identifier reported by the provider, if any.
        role: Response role (typically ``"assistant"``).
        usage: Token usage counts.
        raw: Raw provider payload (kept opaque). Useful for debugging
            without bloating downstream nodes' attention surface.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    role: str = "assistant"
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw: dict[str, Any] | None = None


@MakeNode.from_class(func_name="async_call", outputs={"content": str, "tool_calls": list[dict]})
class BaseLLMNode(BaseNode):
    """Common driver for chat-completion-style LLM providers.

    Subclasses implement :meth:`_complete` (one provider API call) and
    optionally override :meth:`_format_tools_for_provider` if their tool
    schema differs from OpenAI's canonical shape.

    The default :meth:`async_call` runs a bounded tool-calling loop:

    1. Call :meth:`_complete` with ``messages``.
    2. If the response has ``tool_calls`` and a :class:`ToolsNode` is
       bound, dispatch them, append the results to ``messages``, and
       loop.
    3. Stop once the LLM returns no tool calls or
       ``max_tool_iterations`` is reached.

    Subclasses must NOT use ``**kwargs`` on ``async_call`` — the
    framework's schema generator rejects them. Pass per-call provider
    overrides via the ``params: dict`` argument; :meth:`_complete`
    receives them as ``**kwargs``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tools_node: Any = None  # Optional[ToolsNode]; typed Any to avoid cycle.
    max_tool_iterations: int = 3
    # Optional callable invoked after each provider round-trip in the
    # tool-call loop. Return True to stop early, regardless of whether
    # the response still has pending tool calls. Typed as Any to avoid
    # pydantic refusing the Callable shape on `arbitrary_types_allowed`.
    stop_condition: Any = None

    def bind_tools(self, tools: list[BaseNode]) -> BaseLLMNode:
        """Attach a list of tool nodes that the LLM may call.

        Builds a :class:`ToolsNode` internally and stores it; the next
        invocation will surface OpenAI-style tool schemas to the
        provider and dispatch any returned tool calls.
        """
        from llmagpie.base.tools import ToolsNode

        self.tools_node = ToolsNode(name=self.name + "_ToolsNode", tools=tools)
        return self

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """One round-trip to the provider. Subclasses must implement."""
        raise NotImplementedError

    def _format_tools_for_provider(self) -> list[dict] | None:
        """Return tool schemas in the provider's expected shape.

        Default: OpenAI-style ``[{"type": "function", "function":
        {"name", "description", "parameters"}}]``. Override if the
        provider expects a different envelope (Anthropic, Ollama, …).
        """
        if self.tools_node is None:
            return None
        return self.tools_node._generate_openai_schema()

    def _append_tool_messages(
        self,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        tool_outputs: list[dict[str, Any]],
    ) -> None:
        """Append the assistant's tool-call message and each tool's
        response back into the conversation. Default shape mirrors the
        OpenAI chat-completions tool-call protocol; override per
        provider if needed."""
        messages.append(
            {
                "role": response.role,
                "tool_calls": [
                    {"id": ele["id"], "type": ele["type"], "function": ele["function"]}
                    for ele in tool_outputs
                ],
            }
        )
        for ele in tool_outputs:
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(ele.get("output")),
                    "tool_call_id": ele["id"],
                }
            )

    async def async_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        direct_tool_outputs: bool = False,
        params: dict[str, Any] | None = None,
    ):
        """Drive the tool-calling loop. Yields one :class:`LLMResponse`
        per round-trip; the final yield is the terminal response.

        Args:
            model: Provider-specific model identifier.
            messages: OpenAI-style ``[{"role", "content"}]`` history.
            direct_tool_outputs: If True, yield the first response
                (including tool calls) without dispatching the tools.
                The caller is then responsible for handling them.
            params: Optional dict of per-call provider parameters
                forwarded to :meth:`_complete` as ``**kwargs`` —
                temperature, max_tokens, response_format, etc.

        Yields:
            :class:`LLMResponse` after each provider call. The last
            yield has ``tool_calls == []`` (or the loop hit
            ``max_tool_iterations``).
        """
        kwargs = params or {}
        response = await self._complete(model, messages, **kwargs)

        # If the caller-supplied stop_condition fires on the very first
        # response, honor it immediately — even before the tool loop.
        if self.stop_condition is not None and self.stop_condition(response):
            yield response
            return

        if direct_tool_outputs or self.tools_node is None or not response.tool_calls:
            yield response
            return

        iterations = 0
        while iterations < self.max_tool_iterations and response.tool_calls:
            yield response
            tool_result = await self.tools_node.async_call_(tool_calls_list=response.tool_calls)
            tool_outputs = tool_result.get("tool_calls_list", [])
            self._append_tool_messages(messages, response, tool_outputs)
            response = await self._complete(model, messages, **kwargs)
            iterations += 1
            if self.stop_condition is not None and self.stop_condition(response):
                break
        yield response
