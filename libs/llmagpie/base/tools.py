# from collections import OrderedDict
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

# typing
from uuid import uuid4

from pydantic import ConfigDict

from llmagpie.base.node import BaseNode, MakeNode


@MakeNode.from_class(func_name="fire", outputs={"tool_calls_list": list[dict]})
class ToolsNode(BaseNode):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tools: list[BaseNode]
    tools_with_mapping: dict[str, BaseNode]
    max_workers: int = 4

    def __init__(self, *args, **kwargs):
        tools_with_mapping = {ele.name: ele for ele in kwargs.get("tools", [])}

        kwargs["tools_with_mapping"] = tools_with_mapping
        super().__init__(*args, **kwargs)

    def _generate_openai_schema(self):
        return [ele._generate_description_openai() for ele in self.tools]

    def __repr__(self):
        return f"{list(self.tools_with_mapping.keys())}"

    def __str__(self):
        return self.__repr__()

    async def fire(self, tool_calls_list: list[dict]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for _i, ele in enumerate(tool_calls_list):
                if ele.get("function", None):
                    function_args = ele["function"]
                    ele["id"] = ele.get("id", uuid4().hex)

                    # Snapshot the caller's contextvars *per submission*
                    # so tools running on the executor's worker threads
                    # see the active ``RunContext``. A fresh copy per
                    # submission is required: ``Context.run()`` cannot
                    # be entered concurrently on the same Context, so
                    # parallel tool calls need independent Context
                    # objects.
                    ctx = contextvars.copy_context()

                    try:
                        _tool = self.tools_with_mapping[function_args["name"]]
                        args = function_args["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)

                        self.logger.info(f"Running tool: {_tool.name}")
                        future = executor.submit(ctx.run, _tool.run, **args)  # type: ignore[arg-type]

                    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        self.logger.warning(f"Tool dispatch failed for {function_args!r}: {exc}")

                        def _failed(exc: Exception = exc) -> Exception:
                            return Exception(f"Function argument is wrong: {exc}")

                        future = executor.submit(ctx.run, _failed)
                    ele["_f"] = future

            _result = [
                e["_f"].result() if not e["_f"].exception() else e["_f"].exception()
                for e in tool_calls_list
            ]
            for ele, res in zip(tool_calls_list, _result, strict=False):
                ele.update(
                    {
                        "output": res if not isinstance(res, Exception) else None,
                        "error": res if isinstance(res, Exception) else None,
                    }
                )
                ele.pop("_f")

        return tool_calls_list
