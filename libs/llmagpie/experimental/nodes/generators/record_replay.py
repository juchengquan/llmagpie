"""Record/replay fixtures for LLM providers.

In "record" mode, :class:`RecordReplayLLMNode` forwards each
``_complete`` call to its inner provider and saves the
(request, response) pair to a JSON tape file. In "replay" mode it
serves matching pairs from the tape without touching the network.

Pattern: write a test that exercises an agent end-to-end. Run it
once in record mode against a real API. Commit the tape. From then
on CI runs in replay mode — deterministic, free, and offline.

Tape format (newline-delimited JSON, one entry per recorded call)::

    {"request": {"model": ..., "messages": [...], "kwargs": {...}},
     "response": {<LLMResponse.model_dump()>}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from ._base import BaseLLMNode, LLMResponse


def _request_key(model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    """Stable digest of a call signature. Same as cache.py but kept
    local so the two modules can evolve independently."""
    payload = {"model": model, "messages": messages, "kwargs": kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


class TapeMissError(KeyError):
    """Raised in replay mode when an incoming request doesn't match
    any entry in the tape. Carries the unmatched request payload for
    diagnostics — usually because a test changed its prompt or
    parameters but the tape wasn't re-recorded."""

    def __init__(self, message: str, *, request_key: str, request_preview: dict[str, Any]) -> None:
        super().__init__(message)
        self.request_key = request_key
        self.request_preview = request_preview


class RecordReplayLLMNode(BaseLLMNode):
    """Record-or-replay wrapper around any :class:`BaseLLMNode`.

    Args:
        inner: The real provider node to defer to in record mode.
        tape_path: Path to the JSON-lines tape file. Created on first
            write; loaded eagerly on construction (so a typo in the
            path fails loudly rather than silently re-recording).
        mode: ``"replay"`` (default) — never call the inner, raise
            :class:`TapeMissError` on a request not in the tape.
            ``"record"`` — always call the inner and append to the
            tape (overwrites any existing tape on first write).
            ``"auto"`` — replay when the tape exists, record when it
            doesn't. Useful for "first run records, subsequent runs
            replay" CI patterns.

    The tape uses JSON-lines so it's diff-friendly and append-only.
    Each line is ``{"request": ..., "response": ...}``; the request
    is stored fully (model, messages, kwargs) so a human can read
    the tape and see exactly what was exchanged. Matching is by
    SHA-256 of the request signature; multiple identical requests
    return their recorded responses in order.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    inner: Any = Field(default=None, description="The wrapped BaseLLMNode")
    tape_path: Any = None
    mode: Literal["replay", "record", "auto"] = "replay"

    # Runtime bookkeeping — declared as PrivateAttrs so pydantic
    # leaves them out of validation but mypy still sees the attributes.
    _resolved_tape_path: Path | None = PrivateAttr(default=None)
    _entries: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _serve_offsets: dict[str, int] = PrivateAttr(default_factory=dict)
    _effective_mode: str = PrivateAttr(default="replay")

    def model_post_init(self, _ctx: Any) -> None:
        path = Path(self.tape_path) if self.tape_path is not None else None
        self._resolved_tape_path = path
        self._entries = self._load_tape(path)
        self._serve_offsets = {}

        effective = self.mode
        if effective == "auto":
            effective = "replay" if path is not None and path.exists() else "record"
        self._effective_mode = effective

    @staticmethod
    def _load_tape(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
        return entries

    def _next_for_key(self, key: str) -> dict[str, Any] | None:
        offset = self._serve_offsets.get(key, 0)
        matches = [e for e in self._entries if e.get("_key") == key]
        if offset >= len(matches):
            return None
        self._serve_offsets[key] = offset + 1
        return matches[offset]

    def _append(
        self,
        key: str,
        model: str,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        response: LLMResponse,
    ) -> None:
        entry = {
            "_key": key,
            "request": {"model": model, "messages": messages, "kwargs": kwargs},
            "response": response.model_dump(),
        }
        self._entries.append(entry)
        if self._resolved_tape_path is None:
            return
        self._resolved_tape_path.parent.mkdir(parents=True, exist_ok=True)
        with self._resolved_tape_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def bind_tools(self, tools: list[Any]) -> RecordReplayLLMNode:
        self.inner.bind_tools(tools)
        self.tools_node = self.inner.tools_node
        return self

    def _format_tools_for_provider(self) -> list[dict] | None:
        return self.inner._format_tools_for_provider()

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        key = _request_key(model, messages, kwargs)

        if self._effective_mode == "replay":
            entry = self._next_for_key(key)
            if entry is None:
                raise TapeMissError(
                    f"No tape entry matched (key={key}); "
                    "re-record the tape or check the test inputs.",
                    request_key=key,
                    request_preview={"model": model, "messages": messages, "kwargs": kwargs},
                )
            return LLMResponse.model_validate(entry["response"])

        # record / auto-record path: hit the real provider, save the pair.
        response = await self.inner._complete(model, messages, **kwargs)
        self._append(key, model, messages, kwargs, response)
        return response
