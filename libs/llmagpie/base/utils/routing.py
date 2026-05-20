"""Multi-way routing helpers — fan a single source into exactly one of
N downstream branches based on a discrete output value.

The framework already supports binary gating via `cond_func` on a
destination node; this module is sugar for the common N-way case so
callers don't have to set up `cond_func` per branch by hand."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmagpie.base.connectable import BaseConnectable
    from llmagpie.base.pipeline import BasePipeline


def multi_switch(
    pipeline: BasePipeline,
    src: BaseConnectable,
    src_key: str,
    branches: dict[str, BaseConnectable],
    *,
    dest_key: str,
    selector: Callable[[Any], Any] | None = None,
) -> None:
    """Wire ``src``'s ``src_key`` output into exactly one of ``branches``.

    Each branch is mapped to a string ``case_id``. The downstream node
    in each branch will only execute when ``src``'s emitted value (or
    ``selector(value)`` if provided) equals that branch's ``case_id``.

    Mechanically: for each branch, this sets ``branch.cond_func`` to a
    lambda that compares the incoming value against ``case_id``, and
    then calls :meth:`BasePipeline.add_edge` to wire the edge. Existing
    ``cond_func`` on a branch is overwritten — pass branches that don't
    already have one set.

    Args:
        pipeline: The pipeline to wire edges into. Must be uncompiled.
        src: The connectable producing the route value.
        src_key: Key on ``src``'s output that carries the route value.
        branches: Mapping of ``case_id`` → downstream connectable. The
            ``case_id`` strings must be unique.
        dest_key: Input key on each branch that should receive the route
            value (typically the same on every branch — that's why it's
            shared here rather than per-branch).
        selector: Optional pre-processor; called with the value of
            ``src[src_key]`` to produce the value compared against each
            ``case_id``. Useful when ``src`` emits a complex object and
            the route key is a sub-field.

    Example::

        # `router` emits "kind": "weather" | "search" | "chat".
        multi_switch(pipe, router, src_key="kind", dest_key="kind", branches={
            "weather": weather_node,
            "search":  search_node,
            "chat":    chat_node,
        })
    """
    if not branches:
        raise ValueError("branches must be non-empty")
    if len(set(branches.keys())) != len(branches):
        raise ValueError("branches keys must be unique")

    _identity: Callable[[Any], Any] = lambda x: x  # noqa: E731
    sel = selector or _identity

    for case_id, branch in branches.items():
        # Default-arg trick to capture loop vars at definition time, not
        # at call time, so each branch keeps its own case_id.
        def _cond(case: str = case_id, dk: str = dest_key, _sel=sel, **kw: Any) -> bool:
            return _sel(kw[dk]) == case

        branch.cond_func = _cond
        branch.inputs_to_cond = {dest_key: dest_key}
        pipeline.add_edge(src, branch, src_key=src_key, dest_key=dest_key)
