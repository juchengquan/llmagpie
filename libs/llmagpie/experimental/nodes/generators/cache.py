"""Response caching for LLM nodes.

The cache key is computed from ``(model, messages, kwargs)`` so the
same prompt + parameters always hits the same entry. Useful for
deterministic dev/test loops (don't burn API quota retrying the same
call) and for replaying golden conversations.

Two backends ship out of the box: :class:`InMemoryCache` (process-local
dict) and :class:`FileCache` (one file per key in a directory). A
:class:`CacheBackend` protocol lets callers plug in Redis, Memcached,
DynamoDB, etc."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import ConfigDict, Field

from ._base import BaseLLMNode, LLMResponse


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal async key-value interface used by :class:`CachedLLMNode`.

    ``get`` returns the stored bytes (or None on miss / expiry). ``set``
    stores a value with an optional TTL in seconds; ``None`` means "no
    expiry". Implementations should be safe to call concurrently from
    multiple coroutines.
    """

    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None: ...


def _make_key(model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    """Stable cache key over the full call shape.

    Uses sorted-key JSON so dict ordering doesn't change the hash, and
    SHA-256 truncated to 32 hex chars (enough to avoid collisions with
    the worst-case ~100K cached responses).
    """
    payload = {"model": model, "messages": messages, "kwargs": kwargs}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


class InMemoryCache:
    """Process-local, no-eviction cache. Good for tests and dev loops."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[bytes, float | None]] = {}

    async def get(self, key: str) -> bytes | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)


class FileCache:
    """One file per key under ``directory``. Survives across processes.

    Uses an atomic write (tempfile + rename) so a partial write can't
    corrupt the entry. Reads check the mtime-based TTL.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.bin"

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        # ttl ignored for now — file cache is best for "persist forever"
        # use cases. Pluggable cleanup would be a follow-up.
        path = self._path(key)
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as tmp:
            tmp.write(value)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)


class CachedLLMNode(BaseLLMNode):
    """Wraps another :class:`BaseLLMNode` so its ``_complete`` calls
    short-circuit on cache hit.

    Args:
        inner: The LLM node whose ``_complete`` will be cached.
        cache: A :class:`CacheBackend` instance.
        ttl: Optional TTL (seconds) for newly written entries; ``None``
            means "no expiry".

    The cached node inherits ``async_call`` and the tool-calling loop
    from BaseLLMNode unchanged — only the per-call provider round-trip
    is cached. Tool dispatch results are NOT cached (tools may have
    side effects).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    inner: Any = Field(default=None, description="The wrapped BaseLLMNode")
    cache: Any = Field(default=None, description="The CacheBackend instance")
    ttl: int | None = None

    def bind_tools(self, tools: list[Any]) -> CachedLLMNode:
        """Forward tool binding to the inner node so its
        ``_format_tools_for_provider`` sees them; mirror onto self so
        the driver loop's ``self.tools_node`` is also populated."""
        self.inner.bind_tools(tools)
        self.tools_node = self.inner.tools_node
        return self

    def _format_tools_for_provider(self) -> list[dict] | None:
        # Delegate to the inner provider's translation (Anthropic/Ollama
        # may transform the OpenAI-style schema differently).
        return self.inner._format_tools_for_provider()

    async def _complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        key = _make_key(model, messages, kwargs)
        cached_bytes = await self.cache.get(key)
        if cached_bytes is not None:
            return LLMResponse.model_validate_json(cached_bytes)
        response = await self.inner._complete(model, messages, **kwargs)
        await self.cache.set(key, response.model_dump_json().encode(), ttl=self.ttl)
        return response
