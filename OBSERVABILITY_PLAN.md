# Debugging & Observability UX — Plan

> **Status**: planning document for review. No code changed yet.
> **Layers in scope**: `experimental/agent.py` (single-agent) and `experimental/orchestration/` (supervisor/worker). `base/` left alone for now — its errors are already reasonable and the surface is conservative.
> **Workstreams**: error UX, OTel GenAI semantic conventions, structured logging, debug-mode runtime capture.

## Executive summary

llmagpie has the building blocks (`DelegationTrace`, `LLMUsage` rollup, the `opentelemetry_tracer` decorator on `BaseNode._async_execute`), but when something goes wrong at runtime users get a raw async traceback with no session id, no worker name, no trace, and no actionable hint. The opentelemetry decorator emits *generic* `BaseNode` spans — none of them carry GenAI semconv attributes like `gen_ai.usage.input_tokens`, so existing telemetry tools (Phoenix, Arize, LangSmith) can't render the trace tree meaningfully. The logging is hardcoded to Singapore time and doesn't carry correlation IDs across the async boundary.

This plan ships four interlocking workstreams over **four phases** (each phase shippable on its own):

1. **Error UX** — every framework exception carries a `RunContext` attribute, and `format_error()` produces a human-readable post-mortem including the `DelegationTrace`.
2. **OTel GenAI semconv** — three new context-manager helpers (`agent_span`, `handoff_span`, `tool_span`) emit spans matching the OTel GenAI + OpenInference conventions, wired through `Agent.run()`, `WorkerHandle.dispatch()`, and the LLM call path.
3. **Structured logging** — ContextVar-based `run_id` / `depth` / `agent` correlation, optional JSON output, configurable timezone (replaces the hardcoded `Asia/Singapore`).
4. **Debug-mode runtime capture** — `Agent(debug=True)` / `Supervisor(debug=True)` writes a JSONL tape of every LLM round-trip to `debug_dir`, reusing the existing `RecordReplayLLMNode` format.

All four are designed to be *no-ops* when their respective dependencies aren't installed or configured. The default install footprint stays at `networkx + pydantic` and no new core deps are added; the OTel work continues to live under the `[opentelemetry]` extra.

---

## Part 1 — What's broken today

Concrete pain points, drawn from the code that exists today:

**No context on framework exceptions.** Today this is what users see when a supervisor's budget trips two levels deep:

```
BudgetExceededError: supervisor: run exceeded max_tokens_per_run (15234 > 10000)
```

That message includes the supervisor's name but nothing about *which delegation* tripped it, *which worker* was in flight, or *which session* it was. The `DelegationTrace` we just built in PR #13 has all that information — we just don't surface it.

**OTel spans exist but are unusable.** The `opentelemetry_tracer` decorator (`libs/llmagpie/core/opentelemetry/_wrapper.py`) wraps `BaseNode._async_execute` and emits one span per node call with `input.value` and `output.value` JSON-blob attributes. That's the OpenInference v0 convention from 2024. The 2026 OTel GenAI semconv expects:

- Span names: `invoke_agent`, `execute_tool`, `chat`, `handoff`.
- Attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.agent.name`, `gen_ai.tool.name`.
- `openinference.span.kind` set to one of `Agent | Tool | LLM | Chain | Retriever | Reranker | Embedding | Guardrail | Evaluator | Prompt`.

Phoenix, Arize, LangSmith, Langfuse, MLflow all assume these attributes. Without them, the trace tree renders but says "Generic Node" everywhere and has no token-cost rollup.

**Hard-coded Singapore TZ in logging.** `libs/llmagpie/base/logging/logging.py` hardcodes `ZoneInfo("Asia/Singapore")` as the log timestamp timezone. A deployment opinion in framework code — already noted in the abstraction audit as the clearest leftover wart in `base/`.

**No correlation across async boundaries.** When a supervisor's worker runs in a sub-task and logs a warning, the log line has no way to say "this came from supervisor `s1`'s 3rd delegation to worker `researcher` at depth 2." Python's `contextvars` propagate across `await` and `asyncio.create_task` but not across `ThreadPoolExecutor` (which `ToolsNode` uses) — needs explicit threading.

**No way to post-mortem an LLM run.** When a real-provider run fails or produces garbage, users have nothing to inspect. `RecordReplayLLMNode` is close — it captures `(request, response)` pairs to a tape — but it's a wrapper users have to opt into per-LLM-node. A `debug=True` flag on `Agent`/`Supervisor` should do this automatically.

---

## Part 2 — Recommended design

### 2.1 Module layout

```
libs/llmagpie/observability/                # NEW top-level dir
├── __init__.py                             # public re-exports
├── _context.py                             # RunContext + ContextVars
├── _errors.py                              # exception enrichment helpers
├── _format.py                              # format_error() / format_trace()
├── _otel.py                                # agent_span / handoff_span / tool_span
├── _logging.py                             # ContextVar log filter + JSON formatter
└── _capture.py                             # debug-mode JSONL tape capture
```

**Why a top-level `observability/`** (not `experimental/observability/`): unlike provider nodes / Chroma / SQLite, the observability surface is intended to be stable. It's the kind of thing users wire into their app once and rely on. It also touches `base/` (Connectable error paths) and `experimental/` (Agent + orchestration) — putting it in either feels wrong.

The module is *importable on a minimal install* (no `opentelemetry` / `wrapt` dependency at module load) — same lazy-import pattern the rest of the codebase uses. `format_error()`, `RunContext`, and the JSONL capture are pure-stdlib. Only `_otel.py` lazy-imports `opentelemetry` and `wrapt`, and only when an OTel collector is actually configured.

### 2.2 Workstream A — Error UX

Goal: every framework-raised exception carries enough context to debug without a console session and a printf.

**The `RunContext` data class** (in `_context.py`):

```python
@dataclass
class RunContext:
    """Per-run correlation state. Lives in a ContextVar so it propagates
    across `await` and `asyncio.create_task` automatically."""
    run_id: str                              # uuid4().hex — fresh per run() call
    agent: str | None = None                 # innermost agent.name
    supervisor: str | None = None            # outermost supervisor.name
    worker: str | None = None                # current worker, if in dispatch
    depth: int = 0                           # supervisor delegation depth
    thread_id: str | None = None             # memory thread, when applicable
    delegation_trace: Any = None             # DelegationTrace, when supervisor

# Module-level ContextVar; reads are O(1) and always safe (default empty context).
_run_ctx: ContextVar[RunContext | None] = ContextVar("llmagpie_run_ctx", default=None)

def current() -> RunContext | None:
    return _run_ctx.get()

@contextmanager
def push(ctx: RunContext):
    token = _run_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _run_ctx.reset(token)
```

**Exception enrichment** (in `_errors.py`):

```python
def attach_context(exc: BaseException, ctx: RunContext | None = None) -> BaseException:
    """Stash the current RunContext onto an exception as
    `exc.run_context`. Idempotent — if a context is already attached
    (from a deeper frame), preserve it."""
    if getattr(exc, "run_context", None) is None:
        exc.run_context = ctx if ctx is not None else current()
        # Also surface the DelegationTrace if available — pretty-printed
        # via __str__ override below if user calls str(exc).
    return exc
```

Then `Agent.run()`, `Supervisor.run()`, and `WorkerHandle.dispatch()` get a `try/except` that calls `attach_context(exc); raise` — no swallowing, just enrichment. `BudgetExceededError` and `StructuredOutputError` get `run_context: RunContext | None = None` as a new instance attribute (back-compat: it's optional, defaults to None).

**`format_error()`** (in `_format.py`):

```python
def format_error(exc: BaseException, *, include_trace: bool = True) -> str:
    """Human-readable multi-line post-mortem for any exception that
    flowed through llmagpie's runtime.

    Output shape:

        BudgetExceededError: supervisor exceeded max_tokens_per_run (15234 > 10000)
          run_id:     7af3c1b8
          agent:      writer
          supervisor: planner
          worker:     writer (depth 2)
          thread:     thread_alpha

        Delegation trace:
          [1.20s] planner: Write a brief on Mamba SSMs  (15234 tok)  BUDGET EXCEEDED
            [0.40s] researcher: Find authoritative sources  (5100 tok)
            [0.78s] writer: Draft a 200-word summary  (8200 tok)
    """
```

Doesn't replace tracebacks — works alongside them. Callers do:

```python
try:
    result = await supervisor.run("...")
except BudgetExceededError as e:
    print(format_error(e))   # human-readable
    raise                     # let the original traceback continue if needed
```

### 2.3 Workstream B — OTel GenAI semconv

Goal: spans that Phoenix / Arize / LangSmith / Langfuse can render correctly, with token-cost rollup at every level.

**Three context managers** (in `_otel.py`):

```python
@contextmanager
def agent_span(*, agent_name: str, attributes: dict | None = None):
    """Open a span named `invoke_agent` with OpenInference + GenAI attributes:
        openinference.span.kind = "Agent"
        gen_ai.agent.name = agent_name
        (plus caller-supplied attrs)
    No-op when OTel isn't configured (returns a null span object)."""

@contextmanager
def handoff_span(*, source: str, target: str, task: str, depth: int):
    """Span named `handoff` linking supervisor → worker.
        openinference.span.kind = "Chain"
        gen_ai.operation.name = "handoff"
        llmagpie.handoff.source = source
        llmagpie.handoff.target = target
        llmagpie.handoff.depth = depth
        llmagpie.handoff.task_preview = task[:120]"""

@contextmanager
def tool_span(*, tool_name: str):
    """Span named `execute_tool`.
        openinference.span.kind = "Tool"
        gen_ai.tool.name = tool_name"""
```

Plus a thin wrapper around the LLM call path:

```python
def set_llm_attributes(span, *, model: str, usage: LLMUsage,
                       finish_reason: str | None, provider: str | None = None):
    """Stamp gen_ai.* attributes onto the currently-open LLM span.
        gen_ai.system = provider                 # 'openai' / 'anthropic' / 'ollama'
        gen_ai.request.model = model
        gen_ai.usage.input_tokens = usage.prompt_tokens
        gen_ai.usage.output_tokens = usage.completion_tokens
        gen_ai.response.finish_reasons = [finish_reason]"""
```

**Wiring**:

- `Agent.run()` opens an `agent_span` for its own scope.
- `Supervisor.run()` opens an `agent_span(agent_name=self.name)` with `openinference.span.kind = "Agent"`. Each `WorkerHandle.dispatch()` opens a `handoff_span` that becomes the parent of the worker's `agent_span`.
- `BaseLLMNode._complete()` is wrapped — we hook in via a small refactor on the existing `opentelemetry_tracer` decorator that lets specific subclasses contribute extra attributes via a `_span_attrs(self)` hook.
- `ToolsNode.fire()` opens a `tool_span` per dispatched call. Tool errors are recorded on the span via `span.record_exception(...)`.

The existing `opentelemetry_tracer` decorator stays as-is for non-llm nodes. The new helpers are *additive* — they don't break the existing decorator, they layer on top.

**No-op when off.** Every helper detects `OTEL_ENABLED` at call time and returns a stub span. Same pattern as `EmptyWrapDecorator`. Zero overhead in the default install.

### 2.4 Workstream C — Structured logging

Goal: log lines that carry correlation IDs, timezone is configurable, optional JSON output for shipping to log aggregators.

**Singapore TZ fix** (the long-noted wart):

```python
# Old: ZoneInfo("Asia/Singapore") hardcoded.
# New: read LLMAGPIE_LOG_TZ env var; default to UTC; honor IANA names.
_LOG_TZ = ZoneInfo(os.environ.get("LLMAGPIE_LOG_TZ", "UTC"))
```

Documented in CLAUDE.md. Old behavior preserved by setting `LLMAGPIE_LOG_TZ=Asia/Singapore`.

**ContextVar log filter** (in `_logging.py`):

```python
class RunContextFilter(logging.Filter):
    """Inject the current RunContext fields into every log record so
    formatters can reference them."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _run_ctx.get()
        if ctx is not None:
            record.run_id = ctx.run_id
            record.agent = ctx.agent
            record.worker = ctx.worker
            record.depth = ctx.depth
        else:
            record.run_id = "-"
            record.agent = "-"
            record.worker = "-"
            record.depth = 0
        return True
```

The default `LOGGING_FORMAT` in `base/logging/logging.py` gets a small additive update:

```
[%(asctime)s] run_id=%(run_id)s agent=%(agent)s level=%(levelname)s name="%(name)s" msg="%(message)s"
```

Existing format users (who didn't set `run_id` themselves) get `-` placeholders so nothing breaks.

**Optional JSON formatter**:

```python
class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line. Useful for log aggregators
    that want structured fields rather than parsing a custom format."""
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", "-"),
            "agent": getattr(record, "agent", "-"),
            "worker": getattr(record, "worker", "-"),
            "depth": getattr(record, "depth", 0),
        }, default=str)
```

Opt-in via `get_or_create_logger(..., json=True)` or `LLMAGPIE_LOG_JSON=1` env var.

### 2.5 Workstream D — Debug-mode runtime capture

Goal: turn on `debug=True` on an Agent or Supervisor and get a complete record of every LLM call to disk, no extra wiring required.

```python
class Agent:
    def __init__(self, ..., debug: bool = False, debug_dir: str | None = None):
        ...
        if debug:
            tape_path = self._resolve_debug_path(debug_dir)
            # Wrap the innermost LLM in a RecordReplayLLMNode in record mode.
            # This captures every (request, response) pair to a JSONL tape.
            self._llm = self._wrap_with_capture(self._llm, tape_path)
```

The `_wrap_with_capture` walks down the wrapper chain (memory → cache → provider) to the innermost provider and wraps *just that one* — so the cache and memory layers still work, and the tape captures the actual provider exchanges, not cache short-circuits.

Each run gets its own tape: filename is `<debug_dir>/<run_id>.jsonl`. `run_id` comes from the `RunContext` (so the tape file correlates to log entries with the same `run_id`). The tape format is the existing `RecordReplayLLMNode` format — diff-friendly JSONL.

For supervisors, every worker's capture goes to a separate tape (tagged with the worker name in the filename), so post-mortem inspection is per-agent.

**Zero impact when off.** `debug=False` (default) means no wrapping, no tape, no allocations.

### 2.6 Cross-cutting — `RunContext` is the spine

All four workstreams hang off the same `RunContext` ContextVar. That keeps them coherent:

- The exception enrichment reads `current()`.
- The OTel helpers stamp `RunContext.run_id` as the `gen_ai.session.id` span attribute, so traces can correlate to log lines.
- The log filter injects fields from `current()`.
- The debug-mode tape filename is `f"{ctx.run_id}.jsonl"`.

The single `RunContext` ensures users can trace a single user request from log → tape → trace → exception without picking apart correlation IDs by hand.

**ContextVar + ThreadPoolExecutor caveat.** `ToolsNode.fire()` uses `ThreadPoolExecutor.submit()` to dispatch tool calls in parallel. `contextvars` don't propagate to executor threads by default. The fix: wrap each submission with `contextvars.copy_context()` and `ctx.run(...)`. This is a one-line change in `ToolsNode.fire()` and is its own micro-refactor in Phase 1.

---

## Part 3 — Phased implementation plan

### Phase 1 — Core: `RunContext` + error UX + log correlation + Singapore TZ fix

**Goal**: every llmagpie exception carries actionable context; logs carry `run_id` / `agent` / `worker`; timezone is configurable. No OTel work yet.

**Files touched**:
- New: `libs/llmagpie/observability/__init__.py`, `_context.py`, `_errors.py`, `_format.py`, `_logging.py`.
- Modified: `libs/llmagpie/base/logging/logging.py` — TZ from env var; new `RunContextFilter` attached to the default logger; format string gets `run_id` / `agent` placeholders.
- Modified: `libs/llmagpie/base/tools.py` — wrap `executor.submit(_tool.run, ...)` with `contextvars.copy_context().run(...)`.
- Modified: `libs/llmagpie/experimental/agent.py` — `Agent.run()` enters `push(RunContext(...))`; on exception, `attach_context(exc); raise`.
- Modified: `libs/llmagpie/experimental/orchestration/_supervisor.py` — `Supervisor.run()` enters its own context with `delegation_trace`; on exception, attach trace.
- Modified: `libs/llmagpie/experimental/orchestration/_worker.py` — `WorkerHandle.dispatch()` enters a worker-scoped context.
- New: `tests/test_observability.py` — covers ContextVar propagation across `await`, ThreadPoolExecutor propagation, exception enrichment, `format_error()` output shape.

**Public API**:
```python
from llmagpie.observability import RunContext, current_context, format_error, format_trace
```

**Test strategy**:
- Sync test: mock LLM that records `current_context().run_id` in `_complete`; verify it matches the run_id in `RunContext`.
- Cross-thread test: ToolsNode dispatch — verify `_run_ctx.get()` inside the tool returns the same context.
- Cross-task test: supervisor with parallel workers — each worker sees the supervisor's context, plus its own `worker` field.
- Exception enrichment: trigger `BudgetExceededError` from inside a supervisor + worker — assert `exc.run_context.delegation_trace` is the correct tree.
- `format_error()` output: snapshot test against a known scenario.

**Risks / open questions**:
- ContextVar reset on `try/finally` — needs careful placement so we don't reset the context if the caller is `await`ing.
- Should we expose `RunContext` on `AgentResult` / `SupervisorResult` for inspection after a successful run? Recommendation: yes, populate `result.run_context` (idempotent — readonly view).

---

### Phase 2 — OTel GenAI semconv spans

**Goal**: `agent_span`, `handoff_span`, `tool_span`, plus `set_llm_attributes`. Wired through Agent / Supervisor / WorkerHandle / BaseLLMNode / ToolsNode.

**Files touched**:
- New: `libs/llmagpie/observability/_otel.py`.
- Modified: `libs/llmagpie/core/opentelemetry/_wrapper.py` — add a `_span_attrs(self)` hook on `BaseNode` so subclasses can contribute extra attributes (used by `BaseLLMNode` to add `gen_ai.system`).
- Modified: `libs/llmagpie/experimental/nodes/generators/_base.py` — `BaseLLMNode._complete` opens a `chat` span and stamps `gen_ai.*` attributes from the resulting `LLMResponse`.
- Modified: `libs/llmagpie/experimental/agent.py` — `Agent.run()` wraps body in `agent_span`.
- Modified: `libs/llmagpie/experimental/orchestration/_supervisor.py` — `Supervisor.run()` wraps in `agent_span`; tool dispatch wraps in `tool_span` or `handoff_span`.
- Modified: `libs/llmagpie/experimental/orchestration/_worker.py` — `WorkerHandle.dispatch()` opens `handoff_span` around the inner `agent.run()` call.
- Modified: `libs/llmagpie/base/tools.py` — `ToolsNode.fire()` opens `tool_span` per call.
- New: `tests/test_otel.py` — uses an in-memory OTel `InMemorySpanExporter` to assert spans + attributes.

**Test strategy**:
- Configure an in-memory exporter, run a supervisor with two workers, assert the span tree has the right shape (supervisor `Agent` span → handoff spans → worker `Agent` spans → chat spans → tool spans).
- Assert `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` are present on chat spans.
- Assert tree depth matches the `DelegationTrace.depth`.
- No-op test: without `OTEL_COLLECTOR_ENDPOINT`, the span helpers return no-op stubs and never call into `opentelemetry`.

**Risks / open questions**:
- The existing `opentelemetry_tracer` decorator wraps `BaseNode._async_execute` with its own span ("openinference.span.kind=LLM" hardcoded). If we add new spans at higher levels, do we get double spans? Recommendation: deprecate the existing per-node span in favor of the explicit ones; or pin its span-kind to `Chain` so it nests cleanly.
- `gen_ai.system` value for OpenAI-compatible servers (vLLM, LiteLLM, etc.) — recommendation: read from `self.provider_name` if set on the node, fall back to the class name without the `ChatNode` suffix.

---

### Phase 3 — Debug-mode runtime capture

**Goal**: `Agent(debug=True)` / `Supervisor(debug=True)` writes a JSONL tape per run.

**Files touched**:
- New: `libs/llmagpie/observability/_capture.py` — wrap an LLM in `RecordReplayLLMNode(mode="record")` keyed by `run_id`.
- Modified: `libs/llmagpie/experimental/agent.py` — `Agent.__init__` accepts `debug: bool = False`, `debug_dir: str | None = None`. If `debug`, wrap innermost LLM at `run()` time so each run gets its own tape.
- Modified: `libs/llmagpie/experimental/orchestration/_supervisor.py` — propagate `debug` / `debug_dir` to workers automatically.
- New: `tests/test_debug_capture.py` — assert tape exists, parses, contains all the round-trips.

**Test strategy**:
- Run an agent with `debug=True, debug_dir=tmp_path`; verify `tmp_path/<run_id>.jsonl` exists and lines parse as the expected request/response shape.
- Test supervisor: assert per-worker tape files exist with worker-scoped names.
- Performance smoke: debug=False has zero allocations from the capture layer.

**Risks / open questions**:
- Disk size on long runs. Recommendation: document the `debug_dir` knob, suggest log rotation; add an optional `max_tape_size_bytes` knob (errors out rather than silently truncating).

---

### Phase 4 — JSON logging + docs + examples

**Goal**: ship the `JsonFormatter`, document everything, add a debugging-focused example.

**Files touched**:
- `libs/llmagpie/observability/_logging.py` — `JsonFormatter`.
- `libs/llmagpie/base/logging/logging.py` — `get_or_create_logger(..., json: bool = False)` knob; respect `LLMAGPIE_LOG_JSON` env var.
- README — new "Debugging" section showing the `format_error()` flow and OTel setup.
- CLAUDE.md — note about `RunContext` being the spine of observability.
- CHANGELOG.
- New: `_examples/agents/supervisor_with_debugging.py` — supervisor with `debug=True`, intentionally trips a budget, catches the exception, prints `format_error()`, points at the tape file.

**Test strategy**:
- `_examples` integration runner picks up `supervisor_with_debugging.py` automatically.
- JSON formatter unit test: feed a record, assert valid JSON output with all expected fields.

**Risks**: minimal — this phase is polish + docs.

---

## Part 4 — Out of scope

Deferred — each is its own design exercise:

1. **OTel for the framework core (BasePipeline / BaseNode).** The existing per-node generic span is decent enough; rewriting it is in scope only if Phase 2's deprecation question pushes us there. A dedicated audit later.
2. **LangSmith / Phoenix / Langfuse SDK integrations.** They all consume OTel spans, so Phase 2's OTel work implicitly enables them. A first-class adapter per backend would be a separate "integrations" effort.
3. **Cost dashboards / Grafana panels.** OTel attributes are in place after Phase 2; building a dashboard is a downstream user concern, not framework work.
4. **`format_error()` as a CLI tool** for parsing tape files. Possible follow-up; would belong with the (deferred) pipeline serialization + CLI work.
5. **Step-through debugger / pdb integration.** Real interactive debugging of an agent run. Big lift; pushed to a future phase.
6. **Distributed-trace propagation (multi-process / multi-service).** OTel handles this natively once Phase 2 lands; explicit cross-process testing is its own follow-up.

---

## Ready-to-start checklist

Before Phase 1 begins, decisions still yours:

- [ ] **Module location**: `libs/llmagpie/observability/` (top-level, recommended) vs. `libs/llmagpie/experimental/observability/` (signals it can change). Recommendation: top-level — the API surface is small, stable, and orthogonal to the experimental work.
- [ ] **Singapore TZ migration**: just default to UTC, or preserve existing behavior by reading `LLMAGPIE_LOG_TZ` with `Asia/Singapore` as the default until v0.1.0? Recommendation: default to UTC immediately, document the env-var override in CLAUDE.md.
- [ ] **`RunContext` attached to `AgentResult` / `SupervisorResult`?** Recommendation: yes — read-only view of the final state.
- [ ] **`format_error()` in chat — autoinstall a sys.excepthook?** Recommendation: no — too invasive. Make users call it explicitly.
- [ ] **OTel deprecation of the existing per-node span** vs. layered coexistence. Recommendation: keep coexisting in Phase 2; deprecate in a follow-up after a release cycle of feedback.
- [ ] **Debug-mode default `debug_dir`** when caller doesn't specify. Options: `./.llmagpie-debug/` (in cwd, recommended), system temp dir, or require explicit. Recommendation: cwd-relative — easy to spot, easy to `.gitignore`.
- [ ] **Phase 4 example**: ship just one (`supervisor_with_debugging.py`) or also add a standalone Agent debugging example?
- [ ] **PR cadence**: one PR per phase, or one big one? Recommendation: one per phase — Phase 1 alone is shippable and immediately useful.

Once these are settled, I can start Phase 1.
