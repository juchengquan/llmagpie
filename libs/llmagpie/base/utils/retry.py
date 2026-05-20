"""Retry helpers for flaky node bodies (LLM HTTP calls, network I/O).

The decorator form (:func:`with_retry`) is the primary entry point —
wrap any async callable to get bounded retries with exponential
backoff and optional jitter. Pass an explicit predicate to control
which exceptions retry vs propagate immediately."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

T = TypeVar("T")

_logger = logging.getLogger(__name__)


def with_retry(
    *,
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    backoff_factor: float = 2.0,
    max_backoff: float = 30.0,
    jitter: bool = True,
    retry_on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    should_retry: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that retries an async callable on failure with exponential backoff.

    The decorated function is invoked up to ``max_attempts`` times. After
    the n-th failed attempt (0-indexed), execution sleeps for
    ``min(backoff_base * backoff_factor ** n, max_backoff)`` seconds —
    multiplied by a random factor in ``[0.5, 1.0)`` if ``jitter`` is True.

    Args:
        max_attempts: Total attempts including the first. ``max_attempts=1``
            disables retry entirely (one shot).
        backoff_base: Initial sleep in seconds before the first retry.
        backoff_factor: Multiplier applied per attempt (2.0 = 0.5s, 1s, 2s, …).
        max_backoff: Hard cap on the per-attempt sleep.
        jitter: Multiply each sleep by uniform [0.5, 1.0) to spread herds.
        retry_on: Exception type (or tuple of types) that triggers a retry.
            Anything else propagates immediately. Default: ``Exception``,
            so ``KeyboardInterrupt`` / ``SystemExit`` still escape.
        should_retry: Optional predicate; if provided, called with the
            caught exception and only retries when it returns True. Lets
            callers filter by status code, message, etc.

    Returns:
        A decorator that wraps an async function. The wrapped function
        re-raises the last seen exception once attempts are exhausted.

    Example::

        @with_retry(max_attempts=4, retry_on=(httpx.HTTPError,))
        async def call_llm(client, model, messages):
            return await client.chat.completions.create(...)
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1; got {max_attempts}")
    if backoff_base < 0:
        raise ValueError(f"backoff_base must be >= 0; got {backoff_base}")

    def _decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def _wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if should_retry is not None and not should_retry(exc):
                        raise
                    if attempt + 1 >= max_attempts:
                        raise
                    delay = min(backoff_base * (backoff_factor**attempt), max_backoff)
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5
                    _logger.warning(
                        "Retry %d/%d for %s after %.3fs (caught %s: %s)",
                        attempt + 1,
                        max_attempts,
                        getattr(func, "__qualname__", repr(func)),
                        delay,
                        type(exc).__name__,
                        exc,
                    )
                    await asyncio.sleep(delay)
            # Unreachable: either we returned, or the loop re-raised.
            raise RuntimeError("with_retry exhausted without raising") from last_exc

        return _wrapper

    return _decorator


def with_fallback(
    fallback: Callable[..., Awaitable[T]],
    *,
    catch: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that runs ``fallback`` if the primary callable raises.

    The fallback receives the same ``*args, **kwargs`` as the primary.
    Useful for "if the smart model fails, try the cheap one" patterns.
    Compose with :func:`with_retry` for retry-then-fallback semantics::

        @with_fallback(call_cheap_model)
        @with_retry(max_attempts=3)
        async def call_smart_model(...): ...
    """

    def _decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def _wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except catch as exc:
                _logger.warning(
                    "Primary callable %s failed with %s; falling back to %s",
                    getattr(func, "__qualname__", repr(func)),
                    type(exc).__name__,
                    getattr(fallback, "__qualname__", repr(fallback)),
                )
                return await fallback(*args, **kwargs)

        return _wrapper

    return _decorator
