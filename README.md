# LLMagpie

A lightweight Python framework for composing LLM workflows out of small,
reusable nodes wired into a DAG. Nodes can be plain functions, classes,
or LLM clients; pipelines run them asynchronously, stream results, fan
in/out across branches, loop until a condition, and decompose nested
pipelines as a single node.

> Status: alpha. The core (`BaseConnectable` / `BaseNode` / `BasePipeline`)
> is stable enough for examples and small projects. The `experimental/`
> subpackage (Chroma store, sqlite scheduler, OpenAI generator) is
> deliberately not part of the public surface.

## Install

```bash
# from source
poetry install                     # core
poetry install -E opentelemetry    # with OTEL tracing
poetry install -E exp              # with experimental extras (chromadb, sqlalchemy)
```

Requires Python 3.10+.

## Quick start

```python
from llmagpie import MakeNode, BaseNode, BasePipeline

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class Greet(BaseNode):
    async def async_call(self, name: str):
        return {"outputs": f"Hello, {name}!"}

@MakeNode.from_class(func_name="async_call", outputs={"outputs": str})
class Shout(BaseNode):
    async def async_call(self, outputs: str):
        return {"outputs": outputs.upper()}

greet = Greet(name="greet")
shout = Shout(name="shout")

pipe = BasePipeline(name="hello", nodes=[greet, shout])
# wire output "outputs" of `greet` into input "outputs" of `shout`
(greet >> "outputs") >> ("outputs" >> shout)
pipe.compile()

for state in pipe.invoke(inputs={"greet.name": "world"}):
    print(state.value)
# {'outputs': 'Hello, world!'}
# {'outputs': 'HELLO, WORLD!'}
```

More patterns — branching, looping, nested pipelines, conditional emission,
streaming, multi-input merge — live in [`_examples/simple_composition/`](_examples/simple_composition/).

| Pattern | Example |
|---|---|
| Linear chain, two nodes, single input | `single_pipeline_stack.py` |
| Two upstream nodes merging into one downstream | `single_pipeline_merged.py` |
| Fan-out: one parent into multiple branches that re-join | `single_branches.py`, `chained_branches.py` |
| Multiple parents feeding a single child (multi-input) | `multi_input_with_loop.py` |
| Looping back to an earlier node until a condition holds | `single_input_with_loop.py`, `multi_input_with_loop.py` |
| Conditional emission via `cond_func` on the node | `condition_in_node.py`, `condition_in_basenode.py` |
| Allow a step to skip emission without aborting the pipeline | `single_allow_no_emission.py` |
| Async-iterator streaming output between steps | `single_streaming.py` |
| Composing a pipeline as a node inside an outer pipeline | `nested_pipeline.py`, `chained_with_pipe.py`, `chained_nested_pipe.py` |

## Mental model

```
BaseConnectable          (abstract: anything wireable)
├── BaseNode             (a single unit of work)
│   └── @MakeNode.from_class / from_function
└── BasePipeline         (a compiled DAG of connectables)
```

- **Connections** use the `>>` / `<<` operators. `node_a >> "out_key" >> "in_key" << node_b`
  reads "send `out_key` of A into `in_key` of B".
- **State** is per-session, keyed by `session_id`. Each node owns
  `input_state` / `output_state` / `output_history_state` dictionaries
  keyed by session.
- **Execution** is async-first. `invoke()` is a sync wrapper that drives
  the async event loop; `async_invoke()` is the native entry point.
- **Pipelines must be compiled** with `pipe.compile()` before invocation —
  compilation freezes the input/output schema and validates the DAG.

## Tests

```bash
poetry install --with dev
pytest
```

The suite combines unit tests in [`tests/test_basics.py`](tests/test_basics.py)
with example-runner smoke tests in [`tests/test_examples.py`](tests/test_examples.py)
that exercise every file under `_examples/simple_composition/`.

## OpenTelemetry

Set `OTEL_COLLECTOR_ENDPOINT` to enable OTLP HTTP tracing; otherwise the
tracer decorator is a no-op. See `libs/llmagpie/core/opentelemetry/_wrapper.py`.

## License

GPL-3.0-or-later — see [`LICENSE`](LICENSE).
