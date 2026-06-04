"""No-progress detector for the supervisor loop.

Phase 2 termination guard: if the supervisor's last ``N`` LLM
responses emitted no tool calls AND their content is highly similar,
treat the loop as stuck and terminate. Catches the failure mode
where the supervisor LLM keeps generating slightly-different
restatements of the same thing without delegating or finishing.

Uses :func:`difflib.SequenceMatcher` for similarity — no extra
dependency, good enough for the "are these basically the same
sentence?" question. Tunable threshold + window."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..nodes.generators._base import LLMResponse


class NoProgressDetector:
    """Track recent supervisor responses and report when the loop
    looks stuck.

    Args:
        window: How many consecutive responses must look stuck before
            the detector fires. Default 3.
        similarity_threshold: Cosine-style ratio in ``[0, 1]`` above
            which two responses are considered "the same". Default 0.85.
    """

    def __init__(self, window: int = 3, similarity_threshold: float = 0.85) -> None:
        self.window = window
        self.similarity_threshold = similarity_threshold
        self._history: list[str] = []

    def observe(self, response: LLMResponse) -> None:
        """Record a supervisor response. If the response has tool
        calls, reset the history — the supervisor IS making progress."""
        if response.tool_calls:
            self._history.clear()
            return
        self._history.append(response.content or "")
        # Keep only the most recent `window` entries.
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]

    def is_stuck(self) -> bool:
        """True if the last ``window`` responses are pairwise similar
        above ``similarity_threshold``."""
        if len(self._history) < self.window:
            return False
        # Compare each consecutive pair; require ALL to be similar.
        for a, b in zip(self._history, self._history[1:], strict=False):
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio < self.similarity_threshold:
                return False
        return True
