# Multi-Agent Orchestration for `llmagpie` — Research + Plan

> **Status**: planning document for review. No code has been changed.
> **Pattern in scope**: supervisor / worker (a.k.a. "manager", "orchestrator-worker", "hierarchical").
> **Out of scope** for the first round: swarm-style peer handoffs, debate / vote, persistent multi-agent workflows. See Part 4.

## Executive summary

The 2025–2026 consensus across LangGraph, OpenAI Agents SDK, AutoGen / AG2, and CrewAI is that the **supervisor pattern is implemented as "the supervisor LLM emits a tool call that triggers a sub-agent"** — not as a static graph edge, not as a peer handoff, not as a magic field. The frameworks vary in surface API, context-handoff defaults, and concurrency primitives, but they converge on:

1. **Handoff-as-tool-call** (or, equivalently, *agent-as-tool*) as the delegation mechanic.
2. **Static worker registration** at supervisor-construction time (no runtime discovery).
3. **Defense-in-depth termination**: max turns + budget + a pathology guard (e.g. no-progress / depth cap).
4. **Trace-based cost roll-up** via OpenTelemetry/OpenInference, with an in-process accumulator as a backup.
5. **Per-worker context windows are private by default**; what the worker sees of the supervisor's history is a tunable, not a fixed.

`llmagpie`'s existing `Agent` already has the tool-call loop, `LLMUsage` aggregation, `BudgetExceededError`, and `stop_on_*` factories. The cleanest first cut is therefore an **`agent.as_tool()` adapter + a `Supervisor` wrapper** that re-uses `Agent.run()` rather than a parallel orchestrator. This ships value in ~600 lines of code, leverages everything already built, and leaves room for parallel-fanout / streaming / handoff-style routing as additive phases.

---

## Part 1 — Landscape benchmark

### Side-by-side

| Capability | LangGraph (`create_supervisor`) | OpenAI Agents SDK | AutoGen v0.4 / AG2 | CrewAI (`Process.hierarchical`) |
|---|---|---|---|---|
| Supervisor primitive | `create_supervisor(agents, model, ...)` → `StateGraph` | `Agent(handoffs=[...])` *or* `Agent(tools=[w.as_tool(...)])` | `Swarm` / `SelectorGroupChat` / `AutoPattern` | `Crew(process=hierarchical, manager_llm=...)` |
| Delegation mechanic | Tool returns `Command(goto=..., graph=Command.PARENT)` | `transfer_to_*` tool synthesized from `handoffs=`; or `agent.as_tool()` exposed as a regular tool | `transfer_to_*` synthesized from `handoffs=`; or LLM-selected next speaker | Built-in `Delegate work to coworker` tool whose args are validated against a Pydantic schema |
| Worker registration | Static list of compiled `Pregel` graphs | Static `handoffs=` / `tools=` list | Static `participants=` list | Static `agents=` list (manager kept *out* of `agents=`) |
| Worker context | Full `MessagesState` by default; `output_mode` filters return only | Full prior transcript by default; `handoff(...).input_filter` rewrites it; `nest_handoff_history` (beta) collapses to a single summary | Broadcast to all agents (v0.2/AG2); v0.4 same plus `SocietyOfMindAgent` for isolation | Free-text `context` string the manager writes; plus `Task.context=[other_task]` for declarative chains |
| Aggregation | Worker's last (or full) messages appended to shared history; optional `response_format` produces typed `structured_response` | `RunResult.final_output` is typed via `output_type`; `new_items` carry per-step items | `TaskResult.messages` — last message is conventionally the answer; `SocietyOfMindAgent` runs a `response_preparer` over inner messages | `result.tasks_output[-1]` — "the final response is effectively determined by whichever task happens to run last" (widely criticized) |
| Loop cap | `recursion_limit` (default 25); no FINISH sentinel in modern lib | `max_turns` defaulted; `MaxTurnsExceeded` raised | Composable `TerminationCondition` (`MaxMessageTermination \| TextMentionTermination(...)`) | `Agent.max_iter` (25), `max_rpm`, `max_execution_time` |
| Cost roll-up | Per-`AIMessage` `usage_metadata`; LangSmith aggregates | `RunContextWrapper.usage` accumulates across nested `as_tool()` and handoffs in the same run | Per-call `RequestUsage` on responses; no team-level aggregator in v0.4 (regression from v0.2) | `crew.kickoff().token_usage` with per-task breakdowns |
| Parallel delegation | Yes via `parallel_tool_calls=True` + `Send` fan-out | Yes via parallel tool calls (capped by `RunConfig.tool_execution.max_function_tool_concurrency`); handoffs themselves serial | Strictly serial speaker turn-taking; parallel possible only inside a single agent's tool calls | `Task(async_execution=True)` for sibling tasks; no in-round fan-out |
| Streaming | `astream(stream_mode="messages", subgraphs=True)`; **subgraphs=False silently drops chunks** (open issue #226) | First-class: `Runner.run_streamed()` exposes `current_agent` and a live event stream including handoffs | First-class in v0.4 via `model_client_stream=True`; **breaks when same turn produces tool calls** (#6136) | LLM-level stream only; **stream events arrive out-of-order** (#3008); no hierarchical stream API |
| Observability | LangSmith first-class; OTel export; struggles with `Send`-based concurrent worker spans | Built-in tracing on by default; `agent_span`/`handoff_span`/`generation_span`/`tool_span`/`guardrail_span`; processors for Phoenix/W&B/Langfuse/MLflow/Logfire | Built-in OTel in v0.4 (`autogen-core`); AG2 ships `ag2[tracing]` with `instrument_agent` / `instrument_pattern` | Event bus (`CrewAIEventsBus` + `BaseEventListener`) is the seam; AgentOps / Langtrace / MLflow integrate through it |

### What's convergent (do this)

- **Tool-call delegation** is the load-bearing primitive everywhere it works well. Frameworks that tried other approaches (CrewAI's free-text `context` argument, AutoGen v0.2's "every agent sees every message") have well-documented pain points.
- **Static worker registration** at construction time — none of the four production frameworks support runtime discovery, and field reports don't ask for it.
- **Typed context object** (Agents SDK `RunContextWrapper[T]`, LangGraph typed `State`) replaces ad-hoc dicts. Swarm's `context_variables: dict` was the first thing the Agents SDK fixed.
- **In-run usage accumulator** at the supervisor scope, propagated to nested workers. Cost-per-trace is the most-asked-for production metric (Anthropic's research system uses ~15× the tokens of a chat call¹).
- **Defense-in-depth termination**: every framework with a long-running production cohort has converged on combining a max-turns cap with at least one of: budget cap, no-progress detector, or content-match stop. MAST's empirical 1,600-trace study² shows 21.3% of multi-agent failures are termination-related.

### What's divergent (where llmagpie should make its own call)

- **Default context handoff**: LangGraph defaults to *full history*, OpenAI Agents SDK to *full history with opt-in summarization*, Anthropic's research system deliberately shares *almost nothing*³. Cognition's "Don't Build Multi-Agents" post and Anthropic's "How we built it" landed one day apart in June 2025 with opposite recommendations — domain-dependent.
- **Parallel fan-out**: LangGraph's `Send` is the cleanest primitive; AutoGen has no in-round fan-out; the Agents SDK gets it via parallel tool calls but caps `max_function_tool_concurrency`. The right concurrency primitive is `asyncio.TaskGroup` (or AnyIO nursery), **not** `asyncio.gather` — `gather`'s failure semantics are a known footgun in this domain⁴.
- **Aggregation**: every framework leaves this to the user. CrewAI's "last task wins" is the most-criticized default. Worth designing an explicit `aggregation_strategy` knob.
- **Streaming through hierarchy**: nobody has nailed this. LangGraph and AutoGen both have open bugs; Agents SDK is closest but is OpenAI-specific.

¹ Anthropic engineering, "How we built our multi-agent research system" (2025).
² Cemri et al., MAST: "Why Do Multi-Agent LLM Systems Fail?" (arxiv 2503.13657).
³ Anthropic: "each subagent gets a self-contained task description, an output format, and a fresh context window."
⁴ With `return_exceptions=False`, sibling tasks aren't awaited for cleanup; with `True`, failures are silently folded into the list unless every caller checks.

---

## Part 2 — Recommended design for `llmagpie`

### 2.1 Module boundary

**New module**: `libs/llmagpie/experimental/orchestration/`. Files:

- `_supervisor.py` — `Supervisor` class, `SupervisorResult`, `WorkerHandle`.
- `_worker.py` — `Agent.as_worker(...)` adapter (the agent-as-tool wrapper).
- `_aggregation.py` — built-in aggregators (`last`, `all_messages`, `structured_merge`).
- `_trace.py` — `DelegationTrace` data model + tree-printer.
- `__init__.py` — public exports.

**Why a new subdirectory, not adding to `experimental/agent.py`**: the file is already 400+ lines and tightly scoped to single-agent runtime. Mixing orchestration in would couple two layers that should evolve separately. A subdirectory also signals clearly that this is a higher-level composition (workers *contain* agents), and leaves room for `_swarm.py` / `_debate.py` later without further restructuring.

### 2.2 Public API

```python
from llmagpie.experimental.agent import Agent
from llmagpie.experimental.orchestration import Supervisor, WorkerHandle


# 1. Build worker agents normally — they're just Agents.
researcher = Agent(
    llm=openai_node, model="gpt-4o-mini",
    system_prompt="You find authoritative sources on the topic.",
    tools=[web_search_tool],
)
writer = Agent(
    llm=openai_node, model="gpt-4o-mini",
    system_prompt="You write technical summaries grounded in sources.",
)

# 2. Wrap them as workers. `as_worker()` returns a tool the supervisor can call.
research_worker = researcher.as_worker(
    name="researcher",
    description="Use to gather authoritative sources on a topic.",
    context_handoff="task_only",         # see 2.4
)
write_worker = writer.as_worker(
    name="writer",
    description="Use to draft a summary once research is collected.",
    context_handoff="task_only",
)

# 3. Build the supervisor.
supervisor = Supervisor(
    llm=openai_node, model="gpt-4o",
    system_prompt=(
        "You coordinate a research+writing pipeline. Delegate to "
        "`researcher` and `writer`. Produce the final summary."
    ),
    workers=[research_worker, write_worker],
    max_delegations=10,
    max_depth=3,
    max_tokens_per_run=200_000,
    aggregation="structured_merge",   # or "last", "all_messages", or a callable
    response_schema=FinalReport,      # optional Pydantic schema
)

# 4. Run it.
result = await supervisor.run("Write a 300-word summary of Mamba SSMs.")
print(result.content)
print(result.parsed)        # FinalReport(...)
print(result.usage)         # cumulative LLMUsage across supervisor + workers
print(result.trace)         # DelegationTrace — pretty-prints the call tree
```

### 2.3 Composition with existing abstractions

- Each worker is a **real `Agent`** — with its own memory, cache, tools, budget, stop conditions, streaming. No new "WorkerAgent" type. `agent.as_worker(...)` is just a factory for a `WorkerHandle` (a tool spec + a callable).
- `Supervisor` reuses `Agent`'s tool-call loop. Specifically, `Supervisor` **is** an `Agent` subclass — its only added behavior is (a) registering workers as tools, (b) cumulative usage aggregation across worker invocations, (c) the `trace` field, (d) depth tracking. This means: cache, memory, structured outputs, stop conditions, streaming all come "for free" via the base.
- `WorkerHandle.invoke(args, parent_usage, depth)` is the seam where context-handoff strategy, budget propagation, and trace bookkeeping happen. The supervisor's tool-call dispatch routes here.

### 2.4 Handoff design

The handoff is the load-bearing mechanic of the supervisor pattern and gets the most detailed treatment here. **Seven design principles**, in priority order:

1. **Handoff is a tool call — full stop.** Not a graph edge, not an LLM-picked next speaker, not a special syntax. The supervisor's LLM emits `transfer_to_<worker>(...)` like any other tool. Matches what the model already knows how to do, keeps the loop introspectable, makes hallucinations recoverable. LangGraph, the Agents SDK, AutoGen Swarm, and CrewAI all converged here.
2. **Arguments are typed and validated at the boundary.** A `HandoffArgs` Pydantic schema with `task: str` (required, self-contained) and optional fields. Validation rejection is **not** an exception — it's a tool-result message the LLM can retry from. CrewAI's #2606 (their schema crashed when the model passed a dict instead of a string) is exactly what this prevents.
3. **Default context is "task only" — be Anthropic, not LangGraph.** The supervisor passes a self-contained task string; the worker sees that plus its own system prompt. Nothing else by default. Easy to relax (`"task_plus_history"`); impossible to claw back token bloat after the fact.
4. **Workers can't hand back.** A worker returns its result and exits; the supervisor's LLM decides what's next. Bidirectional handoff is a different orchestration class (Swarm), not a knob.
5. **Errors travel as tool results, never as raised exceptions.** Worker crash, budget exceeded, schema invalid, hallucinated worker name — all become a tool-result `{"error": "..."}`. The only thing that propagates upward is the supervisor's own budget. A worker failure should never kill the supervisor's run; the LLM is the recovery loop.
6. **Memory is per-delegation by default.** Each `WorkerHandle.invoke()` opens a fresh thread for the worker's memory store. Workers don't accumulate context across delegations unless `persistent_thread=True` is set. Makes delegations stateless and parallelizable.
7. **Usage propagates eagerly, never lazily.** After every worker returns, its `LLMUsage` is added to the supervisor's accumulator *before* the next supervisor LLM call. Budget checks always see truthful cumulative cost — even mid-delegation. Parallel-fanout phase tightens this: a sibling's overage cancels still-running siblings via `TaskGroup`.

#### The wire format

Each `WorkerHandle` exposes itself to the supervisor's LLM as a single tool:

```python
class HandoffArgs(BaseModel):
    task: str = Field(
        description=(
            "A self-contained task description. The worker has no memory of prior "
            "conversation; everything it needs must be in this field."
        )
    )
    context_hint: str | None = Field(
        default=None,
        description="Optional short context the supervisor wants the worker to keep in mind.",
    )
    expected_fields: list[str] | None = None  # used when worker has a response_schema
```

Tool description is auto-generated as `WorkerHandle.description` + a literal enumeration of the worker's available tools, so the supervisor's LLM knows what each worker can actually do:

```
transfer_to_researcher: Delegate a task to the `researcher` agent. The worker
will return content + structured result. Available tools: web_search, fetch_url.
```

#### What the worker sees on input (the three modes)

**`"task_only"` (default)**
```python
messages = [
    {"role": "system", "content": worker.system_prompt},
    {"role": "user", "content": args.task},
]
```
No view of the supervisor's transcript, no other workers' results.

**`"task_plus_history"`**
```python
messages = [
    {"role": "system", "content": worker.system_prompt},
    # Last N supervisor messages, role-flipped: supervisor's "assistant" turns
    # become user-side annotated context (<supervisor_message>...</supervisor_message>),
    # so the worker treats them as background rather than turns to continue from.
    *_summarize_tail(parent_messages, n=history_window),
    {"role": "user", "content": args.task},
]
```

**`"shared_scratchpad"`**
```python
messages = [
    {"role": "system", "content": worker.system_prompt + SCRATCHPAD_INSTRUCTIONS},
    {"role": "user", "content": (
        f"<task>{args.task}</task>\n"
        f"<scratchpad>{json.dumps(supervisor.scratchpad)}</scratchpad>"
    )},
]
```
Worker returns a structured patch (`{"updates": {...}}`) merged into the shared scratchpad before the next delegation. Forces structured output — best for genuine coordination where free-text returns lose state.

#### Inside `WorkerHandle.invoke()`

```python
async def invoke(
    self, args: HandoffArgs, *, parent_messages, depth, parent_usage, scratchpad,
) -> WorkerResult:
    if depth >= self._max_depth:
        return WorkerResult(worker=self.name, content="",
                            error=f"max_depth exceeded ({depth})")

    messages = self._build_messages(args, parent_messages, scratchpad)

    # Worker drives its own tool loop / memory / cache / budget / stop_condition.
    # We call `_drive(messages)` directly rather than `worker.run(user_message=...)`
    # so the worker doesn't double-persist the synthesized handoff to its memory store.
    try:
        agent_result = await self.agent._drive(messages, depth=depth + 1)
    except BudgetExceededError as e:
        return WorkerResult(worker=self.name, content="", usage=e.usage_so_far,
                            error=f"worker budget exceeded: {e}")
    except Exception as e:                              # narrow this in practice
        return WorkerResult(worker=self.name, content="", error=repr(e))

    parent_usage += agent_result.usage                  # eager rollup, principle #7
    return WorkerResult(
        worker=self.name,
        content=agent_result.content,
        parsed=agent_result.parsed,
        usage=agent_result.usage,
    )
```

The supervisor's tool-dispatch wraps `WorkerResult` into the tool-result message its LLM sees on the next round:

```python
{
    "role": "tool",
    "tool_call_id": tc["id"],
    "content": json.dumps({
        "worker": result.worker,
        "result": result.content,
        "structured": result.parsed,
        "error": result.error,        # null on success
    }, default=str),
}
```

#### Validation and hallucination handling

Two checks at the supervisor's tool-dispatch boundary:

1. **Worker-name validation.** If the LLM hallucinates `transfer_to_legalbot` and there's no such worker, the dispatch returns a tool-result with `{"error": "unknown worker", "available": ["researcher", "writer"]}` so the LLM can retry. Never crashes.
2. **Args validation.** If the model emits `{"task": null}` or malformed JSON, Pydantic rejects, we return the same tool-error shape with the validation message. Worker is never invoked.

#### Memory scoping

Each `WorkerHandle.invoke()` creates a fresh `thread_id` for the worker's memory store, scoped per delegation. Matches "every subagent gets a fresh context window" from the Anthropic post and avoids cross-run leakage. If you genuinely want a worker that maintains cross-delegation memory (a "team historian"), opt in via `WorkerHandle(persistent_thread=True)`.

#### Worker's own tools

Each worker runs its own tool loop internally (it's a real `Agent`). The supervisor doesn't see the worker's tool calls — only the final `WorkerResult.content`/`.parsed`. Worker tool calls appear in the `DelegationTrace` for observability but never bubble up as tool results to the supervisor's LLM. This is what makes the abstraction cleanly nestable: a worker that's itself a `Supervisor` runs its own delegations privately and surfaces only the final result.

### 2.5 Result aggregation

Workers return a structured `WorkerResult`:

```python
class WorkerResult(BaseModel):
    worker: str
    content: str
    parsed: Any | None         # if the worker had a response_schema
    usage: LLMUsage
    error: str | None = None
```

Three built-in aggregation strategies, plus user-callable:

- `"last"` — supervisor's final content is its last LLM message (what most frameworks do today).
- `"all_messages"` — supervisor's final content is a structured concatenation of all worker outputs in delegation order.
- `"structured_merge"` — when `response_schema` is set, the supervisor's tool-call loop is forced (via `stop_on_*` + the existing self-repair) to emit a final message that validates against the schema.
- Callable: `aggregation=lambda supervisor_messages, worker_results: ...` for full control.

### 2.6 Cost & budget roll-up

- `Supervisor._cumulative_usage` (an `LLMUsage`) is incremented after every supervisor LLM call AND after every `WorkerHandle.invoke()` returns. It includes nested supervisors-of-supervisors recursively because each level adds its own usage to its parent's accumulator on return.
- `max_tokens_per_run`, `max_cost_per_run`, `cost_per_1k_tokens` work identically to the existing `Agent` — they're checked after every round-trip via the same `_enforce_budget()` helper, lifted up to `BaseAgent` (a small refactor — see Phase 1).
- A worker can have its own per-worker budget (its own `Agent.max_tokens_per_run`) that's enforced *in addition* to the supervisor's. If a worker raises `BudgetExceededError`, the supervisor sees it as a tool-result error and the LLM can choose to retry on a different worker, summarize partial results, or stop. The supervisor doesn't auto-fail.

### 2.7 Concurrency

- v1 (Phase 1): **serial** worker dispatch. The supervisor's tool-call loop already calls tools serially; we leave it that way.
- v2 (Phase 3): **parallel fan-out** when the supervisor emits multiple `transfer_to_*` tool calls in one assistant message. Implementation: `asyncio.TaskGroup` (Python 3.11+, already required), with `return_exceptions=True` for cleanup but a hard check at the join. Failed siblings get cancelled cooperatively. Bound concurrency with `Supervisor.max_parallel_workers` (default 4, matching `ToolsNode.max_workers`).
- **Not** using `asyncio.gather` per `tianpan.co` analysis — the cleanup semantics are wrong for our domain.

### 2.8 Streaming

`Supervisor.stream(...)` yields `SupervisorChunk` items:

```python
class SupervisorChunk(BaseModel):
    source: Literal["supervisor", "worker"]
    worker: str | None        # set when source == "worker"
    chunk: StreamChunk        # the underlying LLM stream chunk
    event: Literal["start", "delta", "end"] | None  # boundary marker
```

Contract: when the supervisor delegates, it emits a `start` event with the worker name, then forwards each `StreamChunk` from the worker (tagged with source=worker), then an `end` event. The supervisor's own tokens stream as `source=supervisor`. Parallel workers' streams are multiplexed in arrival order.

This sidesteps the LangGraph / AutoGen streaming bugs by making the supervisor explicitly the multiplexer rather than hoping subgraph events flow through transparently.

### 2.9 Stop conditions & depth cap

- `Supervisor.max_delegations` (default 10) — hard cap on total worker invocations per `run()`. Equivalent to LangGraph's `recursion_limit` but worker-call-counted, not graph-step-counted, so it's stable across context-handoff strategies.
- `Supervisor.max_depth` (default 3) — when a worker is itself a `Supervisor`, this caps the nesting. Each `WorkerHandle.invoke()` increments the depth and refuses if it exceeds the cap.
- `Supervisor.stop_condition` — same callable contract as `Agent.stop_condition`; can be combined with `any_of(...)`. Default: `stop_on_finish_reason("stop")`.
- **No-progress detector** (Phase 2 — see Part 3): if the last N supervisor LLM calls produced no tool calls AND the content didn't change meaningfully (cosine similarity > 0.85), terminate.

### 2.10 Failure modes & their mitigations

| Failure | Mitigation in design |
|---|---|
| Hallucinated worker name | Validate against `workers` registry at tool-dispatch time; return a tool-error message listing valid names so the LLM can retry. |
| Infinite supervisor↔worker ping-pong | `max_delegations` + `max_depth` + Phase-2 no-progress detector. |
| Worker crashes (exception) | Caught; surfaced to supervisor as a `WorkerResult(error=...)` tool result; LLM can route around. |
| Worker exceeds its own budget | `BudgetExceededError` becomes a `WorkerResult(error=...)`; supervisor sees and decides. |
| Worker returns unparseable structured output | Worker's existing `response_schema` self-repair runs first; if still invalid, surfaced as error. |
| Supervisor exceeds run budget | `BudgetExceededError` raised at the supervisor level (existing behavior). |
| Manager LLM ignores delegation and tries to do the work itself | Mitigation is prompt-shaped: include "you MUST delegate; do not answer directly until you have at least one worker result" guidance in default supervisor system-prompt template. Same lesson as CrewAI Discussion #1220. |

### 2.11 Observability

- Each delegation captured in `DelegationTrace`:
  ```python
  class DelegationTrace(BaseModel):
      worker: str
      task: str
      depth: int
      started_at: float
      ended_at: float | None
      usage: LLMUsage
      error: str | None = None
      children: list[DelegationTrace] = []  # nested supervisor calls
  ```
- `SupervisorResult.trace` is the root `DelegationTrace`. `trace.format()` pretty-prints as an indented tree.
- When the OTel decorator is active, `WorkerHandle.invoke()` opens a span with attributes `gen_ai.agent.name=worker.name`, `gen_ai.operation.name=invoke_agent`, `llmagpie.supervisor.depth=N` — aligns with the OTel GenAI semantic conventions.

---

## Part 3 — Phased implementation plan

### Phase 1 — Minimal supervisor (workers-as-tools, serial)

**Goal**: A working `Supervisor` that delegates to worker `Agent`s via tool calls, cumulates usage, enforces budget. No parallelism, no streaming, no advanced context handoff yet.

**Files touched**:
- New: `libs/llmagpie/experimental/orchestration/__init__.py`, `_supervisor.py`, `_worker.py`, `_trace.py`.
- New: `tests/test_orchestration.py`.
- Small refactor: `libs/llmagpie/experimental/agent.py` — lift `_enforce_budget()` and the cost-helper logic up into a small `_BaseAgent` mixin (or just module-level functions) so `Supervisor` can reuse them without inheritance fragility.
- New example: `_examples/agents/supervisor_basic.py`.

**New classes / signatures**:
```python
class WorkerHandle(BaseModel):
    name: str
    description: str
    agent: "Agent"
    context_handoff: Literal["task_only", "task_plus_history", "shared_scratchpad"] = "task_only"
    history_window: int = 6

    async def invoke(
        self, task: str, *, parent_messages: list[dict], depth: int,
        parent_usage: LLMUsage, scratchpad: dict | None = None,
    ) -> "WorkerResult": ...


class WorkerResult(BaseModel):
    worker: str
    content: str
    parsed: Any | None
    usage: LLMUsage
    error: str | None = None


class DelegationTrace(BaseModel):
    worker: str
    task: str
    depth: int
    started_at: float
    ended_at: float | None
    usage: LLMUsage
    error: str | None = None
    children: list["DelegationTrace"] = Field(default_factory=list)

    def format(self, indent: int = 0) -> str: ...


class Supervisor(Agent):  # subclass of Agent — gets run()/stream()/tool loop
    workers: list[WorkerHandle]
    max_delegations: int = 10
    max_depth: int = 3
    aggregation: Literal["last", "all_messages", "structured_merge"] | Callable = "last"

    async def run(self, user_message: str, ...) -> "SupervisorResult": ...


class SupervisorResult(AgentResult):
    trace: DelegationTrace
    worker_results: list[WorkerResult]
```

And on `Agent`:
```python
def as_worker(self, name: str, description: str, **kwargs) -> WorkerHandle: ...
```

**Test strategy** (`tests/test_orchestration.py`):
- Happy path: supervisor delegates to one worker, gets result, produces final answer.
- Multi-worker: delegates to two workers in sequence, supervisor sees both results.
- Cumulative usage: assert supervisor's `result.usage` equals supervisor LLM tokens + sum of worker token usages.
- Budget cap: supervisor with low `max_tokens_per_run` raises `BudgetExceededError` mid-delegation.
- Worker crash: worker raises, surfaces as `WorkerResult(error=...)`, supervisor sees it and finishes.
- Hallucinated worker name: validates against registry, returns actionable error to LLM.
- `max_delegations` cap: supervisor that wants to keep delegating gets stopped.
- `max_depth` cap: supervisor-of-supervisor refuses at depth N+1.
- Regression for `Agent`: re-run a subset of `test_agent.py` to confirm the `_enforce_budget` refactor didn't break anything.

**Risks / open questions**:
- Subclassing `Agent` vs. composition: subclass keeps `run()`/`stream()`/tool-loop free, but conflicts if `Agent` ever gets fields like `tools` that the supervisor wants to manage. Composition is safer long-term. Recommendation: subclass for Phase 1, revisit if friction shows up.
- The `_enforce_budget` refactor — needs to keep existing `Agent` tests passing. Should be a 30-line move.

---

### Phase 2 — Context handoff strategies + no-progress guard

**Goal**: Make context handoff a first-class knob; add the pathology-based termination guard.

**Files touched**:
- `libs/llmagpie/experimental/orchestration/_worker.py` — implement `task_only` / `task_plus_history` / `shared_scratchpad`.
- `libs/llmagpie/experimental/orchestration/_supervisor.py` — add no-progress detector.
- `tests/test_orchestration.py` — extend.
- New example: `_examples/agents/supervisor_handoff_modes.py`.

**New / changed**:
- `WorkerHandle.context_handoff` activated (Phase 1 has only `task_only`).
- `Supervisor.no_progress_threshold: int = 3` — terminate if N consecutive supervisor messages have no tool calls and content cosine-similarity > 0.85.
- A small `_similarity.py` helper (no new deps — `difflib.SequenceMatcher` is fine for v1).

**Test strategy**: deterministic tests with `MockLLMNode` that emits a fixed sequence of "delegate / report / delegate / report" with identical content; assert no-progress detector fires at N. Tests for each context-handoff mode that assert the worker actually saw the right slice of messages.

**Risks**: similarity threshold is a magic number. Document it as tunable.

---

### Phase 3 — Parallel fan-out + structured aggregation

**Goal**: Supervisor can dispatch multiple workers in one assistant turn; structured aggregation produces a validated final output.

**Files touched**:
- `libs/llmagpie/experimental/orchestration/_supervisor.py` — `asyncio.TaskGroup`-based parallel dispatch.
- `libs/llmagpie/experimental/orchestration/_aggregation.py` — `last` / `all_messages` / `structured_merge` implementations.
- Tests.
- New example: `_examples/agents/supervisor_parallel.py`.

**New / changed**:
- `Supervisor.max_parallel_workers: int = 4`.
- When the supervisor's LLM emits multiple `transfer_to_*` tool calls in one assistant message, dispatch them concurrently. Honor `max_parallel_workers` via a semaphore inside the TaskGroup.
- Aggregation strategies plumbed into `Supervisor.run()`'s finalization step.

**Test strategy**:
- Two-worker concurrent dispatch — assert both ran, total wall-clock < sum of individual durations (with deterministic mocks that `await asyncio.sleep(0.01)`).
- Cancel-siblings on failure — one worker raises mid-flight, sibling gets `CancelledError` and partial usage is still captured in the trace.
- Structured aggregation: `response_schema=...` with `aggregation="structured_merge"` produces a validated final.

**Risks**: pure-mock tests of concurrency can be flaky; use the existing async-test patterns (no real `sleep`, just `asyncio.Event` synchronization).

---

### Phase 4 — Streaming + OTel + docs

**Goal**: hierarchical streaming, OTel spans, README & CHANGELOG, real-world example.

**Files touched**:
- `_supervisor.py`: `Supervisor.stream()` yielding `SupervisorChunk`.
- `libs/llmagpie/core/opentelemetry/_wrapper.py`: add a thin `agent_span()` helper used by `WorkerHandle.invoke()`.
- README: new "Supervisor / multi-agent" section.
- CHANGELOG.
- New example: `_examples/agents/supervisor_research_pipeline.py` (research→write→review pipeline with parallel research workers).

**Test strategy**:
- Stream events: assert ordering — `start(researcher) → researcher chunks → end(researcher) → supervisor synthesis chunks`.
- Parallel-worker stream multiplexing: events from two workers interleave correctly.
- OTel: when `OTEL_COLLECTOR_ENDPOINT` is set in a test fixture, assert spans are emitted with the right attributes (mock exporter).

**Risks**: streaming tests are notoriously fragile. Use a `MockLLMNode` that supports `stream_complete` (we already have one in `_examples/agents/_mock.py`).

---

## Part 4 — Out of scope / future work

Deliberately deferred — each is its own design exercise:

1. **Peer handoffs / swarm pattern.** When agents pass control to each other without a central supervisor (OpenAI Swarm style). Workable as a `Swarm` class in the same `orchestration/` module once the supervisor lands, but the control-flow shape is different enough that it shouldn't be bolted on now.
2. **Multi-agent debate / vote.** Multiple workers produce candidate outputs; an aggregator/judge picks or merges. Useful but adds 3-5× cost — needs strong eval motivation before shipping.
3. **Persistent multi-agent workflows.** Durable execution across process restarts (Temporal / Inngest-style). Requires a serializable state model — depends on persistence work that doesn't exist yet.
4. **Vector-memory-aware delegation.** Supervisor uses semantic search over past worker results to decide who to call. Premature; address after we see real usage patterns.
5. **Dynamic worker registration.** None of the four frameworks studied support it and none of the field reports asked for it. Skip unless a real use case shows up.
6. **Cross-process / cross-host workers.** Worker as an RPC endpoint. Belongs to deployment infrastructure, not the framework.

---

## Ready-to-start checklist

Before starting Phase 1, the following decisions are yours:

- [ ] **Subclass vs. composition** for `Supervisor`. Recommendation: subclass `Agent` for Phase 1; revisit if friction appears. Confirm or override.
- [ ] **Default context handoff mode**. Recommendation: `"task_only"`. Aggressive but matches Anthropic's empirical finding and avoids the LangGraph default's token bloat. Confirm or override.
- [ ] **Default `max_delegations` and `max_depth`**. Recommendation: 10 and 3. Confirm or override.
- [ ] **Default `aggregation`**. Recommendation: `"last"` (matches user expectations from chat-style agents). Confirm or override.
- [ ] **`as_worker()` on `Agent` vs. a free function**. Recommendation: method on `Agent` (`my_agent.as_worker(...)`) for ergonomics. Confirm or override.
- [ ] **`Supervisor` lives in `experimental/orchestration/`** (not `experimental/agent.py`). Confirm or override.
- [ ] **Phase ordering OK?** Or want streaming earlier? Or parallelism in Phase 1?
- [ ] **Skip Phase 2's no-progress detector?** It uses `difflib.SequenceMatcher` similarity — simple but somewhat hand-wavy. Acceptable, or want a different heuristic (e.g. tool-call-count-based only)?
- [ ] **Examples**: keep the 4 proposed (`supervisor_basic`, `supervisor_handoff_modes`, `supervisor_parallel`, `supervisor_research_pipeline`) or trim?
- [ ] **PR cadence**: one PR per phase, or one big PR at the end? Recommendation: one PR per phase — Phase 1 alone is shippable.

Once these are settled I can start Phase 1.
