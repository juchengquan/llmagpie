"""OpenAI Chat Completions provider built on :class:`BaseLLMNode`.

This is the recommended OpenAI integration; it composes cleanly with
:class:`Agent`, :class:`MemoryNode`, :class:`CachedLLMNode`, and the
structured-output / streaming infrastructure.

The legacy class ``OpenAIChatCompletionWithToolCall`` in
``openai.py`` is kept for backwards compatibility but isn't getting
new features."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from ._base import BaseLLMNode, LLMResponse, LLMUsage, StreamChunk


class OpenAIChatNode(BaseLLMNode):
    """OpenAI Chat Completions wrapped as a :class:`BaseLLMNode`.

    Args (constructor):
        api_key: OpenAI API key. Defaults to ``OPENAI_API_KEY`` env.
        base_url: Override the API endpoint (e.g. an OpenAI-compatible
            local server: vLLM, Ollama's OpenAI shim, LiteLLM, etc.).
        timeout: HTTP timeout in seconds.
        ssl_verify: Pass-through to httpx.

    Per-call kwargs forwarded to the provider via :meth:`_complete`:
        ``temperature``, ``top_p``, ``max_tokens``, ``stop``,
        ``response_format`` (for native JSON mode), and any other
        Chat-Completions parameter.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    client: Any = None

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        ssl_verify: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            from httpx import AsyncClient
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "Could not import the `openai` package. Install with "
                "`pip install llmagpie[openai]` or `pip install openai httpx`."
            ) from e
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "http_client": AsyncClient(verify=ssl_verify, timeout=timeout),
        }
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        kwargs["client"] = AsyncOpenAI(**client_kwargs)
        super().__init__(*args, **kwargs)

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        call_kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if (tools := self._format_tools_for_provider()) is not None:
            call_kwargs["tools"] = tools
        call_kwargs.update(kwargs)

        completion = await self.client.chat.completions.create(**call_kwargs)
        choice = completion.choices[0]
        msg = choice.message

        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = getattr(tc.function, "arguments", "") or ""
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type or "function",
                        "function": {"name": tc.function.name, "arguments": args},
                    }
                )

        usage = LLMUsage()
        if completion.usage is not None:
            usage.prompt_tokens = completion.usage.prompt_tokens or 0
            usage.completion_tokens = completion.usage.completion_tokens or 0
            usage.total_tokens = completion.usage.total_tokens or 0

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=completion.model,
            role=msg.role or "assistant",
            usage=usage,
            raw=None,
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        """Stream OpenAI's Chat Completions response as :class:`StreamChunk` deltas.

        OpenAI emits one ``ChatCompletionChunk`` per server-sent
        event. Each chunk's ``choices[0].delta`` carries content +
        optional tool_call fragments; the terminal chunk has
        ``choices[0].finish_reason`` set. Pass
        ``stream_options={"include_usage": True}`` (forwarded via
        ``kwargs``) to also receive a final usage chunk.
        """
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if (tools := self._format_tools_for_provider()) is not None:
            call_kwargs["tools"] = tools
        call_kwargs.update(kwargs)

        first_chunk = True
        async for chunk in await self.client.chat.completions.create(**call_kwargs):
            if not chunk.choices:
                # Could be the usage-only final chunk when
                # `stream_options={"include_usage": True}` is set.
                usage_payload = getattr(chunk, "usage", None)
                if usage_payload is not None:
                    yield StreamChunk(
                        usage=LLMUsage(
                            prompt_tokens=usage_payload.prompt_tokens or 0,
                            completion_tokens=usage_payload.completion_tokens or 0,
                            total_tokens=usage_payload.total_tokens or 0,
                        ),
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            tool_calls: list[dict[str, Any]] = []
            for tc in getattr(delta, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                tool_calls.append(
                    {
                        "id": tc.id or "",
                        "type": tc.type or "function",
                        "function": {
                            "name": (fn.name if fn and fn.name else ""),
                            "arguments": (fn.arguments if fn and fn.arguments else ""),
                        },
                    }
                )

            yield StreamChunk(
                delta_content=delta.content or "",
                delta_tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                model=chunk.model if first_chunk else None,
                role=delta.role if first_chunk else None,
            )
            first_chunk = False
