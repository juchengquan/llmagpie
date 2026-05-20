"""Anthropic Messages API provider.

Wires `anthropic.AsyncAnthropic` into the framework's
:class:`BaseLLMNode` abstraction. Translates between OpenAI-style
messages/tool-calls (which the framework uses internally) and
Anthropic's native message + tool_use / tool_result shapes."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import ConfigDict

from ._base import BaseLLMNode, LLMResponse, LLMUsage

try:
    from anthropic import AsyncAnthropic

    _ANTHROPIC_INSTALLED = True
except ImportError:
    _ANTHROPIC_INSTALLED = False


def _split_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Anthropic takes `system` as a top-level kwarg, not a role.
    Split OpenAI-style messages accordingly."""
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
        else:
            rest.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style assistant/tool messages into Anthropic's
    tool_use / tool_result content blocks."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in ("user", "assistant"):
            if m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    args = fn["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        elif role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": str(m.get("content", "")),
                        }
                    ],
                }
            )
    return out


class AnthropicChatNode(BaseLLMNode):
    """Anthropic Messages API node.

    Provider-side state (the SDK client) lives on the instance; per-call
    state (messages, model name) is passed through ``async_call``.

    Args (constructor):
        api_key: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY`` env.
        base_url: Override the API endpoint (e.g. a proxy).
        timeout: HTTP timeout in seconds.

    Per-call kwargs forwarded to the provider via :meth:`_complete`:
        ``max_tokens`` (default 1024), ``temperature``, ``top_p``,
        ``stop_sequences``, ``system`` (auto-extracted from messages).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    client: Any = None
    default_max_tokens: int = 1024

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _ANTHROPIC_INSTALLED:
            raise ImportError(
                "Could not import the `anthropic` package. Install with "
                "`pip install llmagpie[anthropic]` or `pip install anthropic`."
            )
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        kwargs["client"] = AsyncAnthropic(**client_kwargs)
        super().__init__(*args, **kwargs)

    def _format_tools_for_provider(self) -> list[dict] | None:
        """Anthropic tools take ``{name, description, input_schema}``,
        not OpenAI's ``{type: function, function: {name, description,
        parameters}}``. Translate here."""
        if self.tools_node is None:
            return None
        openai_tools = self.tools_node._generate_openai_schema()
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in openai_tools
        ]

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        system, rest = _split_system(messages)
        anthropic_messages = _to_anthropic_messages(rest)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", self.default_max_tokens),
        }
        if system is not None:
            call_kwargs["system"] = system
        if (tools := self._format_tools_for_provider()) is not None:
            call_kwargs["tools"] = tools
        call_kwargs.update(kwargs)

        msg = await self.client.messages.create(**call_kwargs)

        content_text = ""
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content_text += block.text
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", uuid.uuid4().hex),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )

        usage = LLMUsage(
            prompt_tokens=getattr(msg.usage, "input_tokens", 0) if msg.usage else 0,
            completion_tokens=getattr(msg.usage, "output_tokens", 0) if msg.usage else 0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=getattr(msg, "stop_reason", None),
            model=getattr(msg, "model", model),
            role="assistant",
            usage=usage,
            raw=None,
        )
