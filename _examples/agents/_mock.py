"""Inline mock LLM used by the runnable examples.

Real provider classes (``AnthropicChatNode``, ``OpenAINode``,
``OllamaChatNode``) need API keys or a local server — fine for an end
user but they'd make these examples impossible to run in CI. The
mock returns canned responses driven by a script so each example
script demonstrates the framework's wiring without external state."""

from collections.abc import Iterable
from typing import Any

from llmagpie.experimental.nodes.generators._base import BaseLLMNode, LLMResponse, LLMUsage


class MockLLMNode(BaseLLMNode):
    """A BaseLLMNode that ignores its input and returns canned responses
    in order. Once the script is exhausted it loops on the last entry
    so demos don't crash if the user runs more turns than scripted."""

    def configure(
        self,
        script: Iterable[LLMResponse | str],
        *,
        usage_per_call: LLMUsage | None = None,
    ) -> "MockLLMNode":
        """Wire up the canned response sequence.

        Strings are wrapped as ``LLMResponse(content=str)`` for
        convenience. Returns self for fluent chaining.
        """
        prepared: list[LLMResponse] = []
        default_usage = usage_per_call or LLMUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        for item in script:
            if isinstance(item, str):
                prepared.append(LLMResponse(content=item, usage=default_usage))
            else:
                if item.usage.total_tokens == 0:
                    item.usage = default_usage
                prepared.append(item)
        object.__setattr__(self, "_script", prepared)
        return self

    async def _complete(self, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        script = getattr(self, "_script", None) or [LLMResponse(content="(no scripted response)")]
        if len(script) > 1:
            return script.pop(0)
        return script[0]  # keep returning the last scripted entry
