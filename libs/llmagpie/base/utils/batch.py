"""Concurrent batch helpers — fan multiple inputs through a single
compiled pipeline (or node) without the caller having to manage tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmagpie.base.connectable import BaseConnectable
    from llmagpie.base.utils.state import StateResponse


async def async_batch_invoke(
    connectable: BaseConnectable,
    inputs_list: Sequence[dict],
    *,
    max_concurrency: int | None = None,
    return_exceptions: bool = False,
) -> list[list[StateResponse] | BaseException]:
    """Run ``connectable.async_invoke`` once for each entry in ``inputs_list``.

    Each invocation gets its own auto-generated ``session_id`` so the
    per-session state for one entry can't leak into another. Results
    preserve the input order.

    Args:
        connectable: A compiled pipeline or a node.
        inputs_list: Each element is an ``inputs`` dict shaped exactly as
            you'd pass to ``connectable.async_invoke(inputs=...)``.
        max_concurrency: Cap on how many invocations run at once. ``None``
            (default) lets ``asyncio`` schedule them all; pass an int for
            backpressure against rate-limited downstreams.
        return_exceptions: If True (default False), exceptions from a
            single entry are captured into the result list at that
            position instead of aborting the batch.

    Returns:
        A list with one entry per input. Each entry is either a list of
        StateResponse objects (one per yielded state from that invocation)
        or — if ``return_exceptions`` — the exception that aborted it.

    Example:
        >>> outs = await async_batch_invoke(pipe, [{"greet.name": "world"},
        ...                                       {"greet.name": "magpie"}])
        >>> [last.value for last in (o[-1] for o in outs)]
        [{'outputs': 'HELLO, WORLD!'}, {'outputs': 'HELLO, MAGPIE!'}]
    """
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def _one(inputs: dict) -> list[StateResponse]:
        async def _drive() -> list[StateResponse]:
            gen = await connectable.async_invoke(inputs=inputs)
            return [state async for state in gen]

        if sem is None:
            return await _drive()
        async with sem:
            return await _drive()

    coros = [_one(i) for i in inputs_list]
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


def batch_invoke(
    connectable: BaseConnectable,
    inputs_list: Sequence[dict],
    *,
    max_concurrency: int | None = None,
    return_exceptions: bool = False,
) -> list[list[StateResponse] | BaseException]:
    """Sync wrapper around :func:`async_batch_invoke`.

    Drives the asyncio event loop via :func:`asyncio.run`. Don't call
    this from inside a running event loop — use the async version.
    """
    return asyncio.run(
        async_batch_invoke(
            connectable,
            inputs_list,
            max_concurrency=max_concurrency,
            return_exceptions=return_exceptions,
        )
    )
