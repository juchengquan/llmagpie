"""Ollama chat-completion provider (local LLMs via HTTP).

Talks to a running Ollama server (default ``http://localhost:11434``)
using its ``/api/chat`` endpoint. Uses ``httpx`` directly (no provider
SDK required). Install via ``pip install llmagpie[ollama]`` which
pulls in `httpx`."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import ConfigDict

from ._base import BaseLLMNode, LLMResponse, LLMUsage, StreamChunk


class OllamaChatNode(BaseLLMNode):
    """Local-model node backed by Ollama's chat endpoint.

    Args (constructor):
        base_url: Ollama server URL. Defaults to ``http://localhost:11434``.
        timeout: HTTP timeout in seconds.

    Per-call kwargs forwarded to Ollama via :meth:`_complete`:
        ``options`` (a dict — ``{"temperature": …, "top_p": …,
        "num_ctx": …, "num_predict": …}``), ``format`` (``"json"``
        to force JSON output), and any other Ollama-supported field.

    Tool calls are surfaced if the model emits them (Ollama uses
    OpenAI-style ``tool_calls`` in its response, so no translation is
    needed in either direction).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    base_url: str = "http://localhost:11434"
    client: Any = None

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "Could not import the `httpx` package. Install with "
                "`pip install llmagpie[ollama]` or `pip install httpx`."
            ) from e
        # Ollama can be slow on first load; default timeout is more generous.
        kwargs["base_url"] = base_url
        kwargs["client"] = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        super().__init__(*args, **kwargs)

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if (tools := self._format_tools_for_provider()) is not None:
            payload["tools"] = tools
        payload.update(kwargs)

        resp = await self.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        body = resp.json()

        msg = body.get("message", {}) or {}
        content = msg.get("content", "") or ""
        raw_tool_calls = msg.get("tool_calls") or []
        # Ollama returns tool_calls as [{"function": {"name", "arguments": {...}}}, ...]
        # without ids. Normalize to OpenAI-style with an id.
        tool_calls: list[dict[str, Any]] = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args)
            tool_calls.append(
                {
                    "id": tc.get("id", uuid.uuid4().hex),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                }
            )

        usage = LLMUsage(
            prompt_tokens=body.get("prompt_eval_count", 0) or 0,
            completion_tokens=body.get("eval_count", 0) or 0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=body.get("done_reason"),
            model=body.get("model", model),
            role=msg.get("role", "assistant"),
            usage=usage,
            raw=None,
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        """Stream Ollama's ``/api/chat`` response as :class:`StreamChunk` deltas.

        Ollama returns one JSON object per line. Each line carries a
        ``message.content`` slice (the new tokens) plus a ``done``
        boolean; the final line additionally has ``done_reason``,
        ``prompt_eval_count``, and ``eval_count``."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if (tools := self._format_tools_for_provider()) is not None:
            payload["tools"] = tools
        payload.update(kwargs)

        first_chunk = True
        async with self.client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                body = json.loads(line)
                msg = body.get("message", {}) or {}
                delta_role = msg.get("role") if first_chunk else None
                first_chunk = False

                tool_calls: list[dict[str, Any]] = []
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {}) or {}
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": tc.get("id", uuid.uuid4().hex),
                            "type": "function",
                            "function": {"name": fn.get("name", ""), "arguments": args},
                        }
                    )

                usage: LLMUsage | None = None
                if body.get("done"):
                    usage = LLMUsage(
                        prompt_tokens=body.get("prompt_eval_count", 0) or 0,
                        completion_tokens=body.get("eval_count", 0) or 0,
                    )
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

                yield StreamChunk(
                    delta_content=msg.get("content", "") or "",
                    delta_tool_calls=tool_calls,
                    finish_reason=body.get("done_reason"),
                    model=body.get("model"),
                    role=delta_role,
                    usage=usage,
                )

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        await self.client.aclose()
