# live‑crew v1.0 – Canonical Specification

If anything conflicts with your work, **this spec wins – ask before deviating.**

> **Installation quickstart:**
>
> ```bash
> # Create isolated environment & install project + dev extras (reads pyproject.toml)
> uv venv .venv --python 3.13
> uv sync --all-extras --dev
> ```
>
> Python 3.13 is mandatory. No `requirements.txt` files are used – **uv** reads dependencies from `pyproject.toml` and locks them into `uv.lock`. Ask before changing this workflow.

---

## 0  Environment & Tool Versions

| Component                    | Version / Rule                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Python runtime               | **3.13** (CPython)                                                                                                             |
| Package manager              | [`uv`](https://github.com/astral-sh/uv) — dependencies pinned with *compatible version* (`~=`) specifiers in `pyproject.toml`. |
| First‑time setup (one‑liner) | `uv venv .venv --python 3.13 && uv sync --all-extras --dev`                                                                    |
| Pydantic                     | `~=2.5`                                                                                                                        |
| nats‑py                      | `~=2.5`                                                                                                                        |
| Docker Compose spec          | **version 3**                                                                                                                  |
| Vector store (optional)      | Qdrant `~=1.9`; Milvus & Pinecone via adapter                                                                                  |
| Embedding model              | `text‑embedding‑3‑small` **d = 1536**                                                                                          |

### 0.1  uv Development Commands

| Task                     | Command                                                                   |
| ------------------------ | ------------------------------------------------------------------------- |
| **Run test suite**       | `uv run -- pytest -q`                                                     |
| **Lint & autofix**       | `uvx ruff check . --fix`                                                  |
| **Type‑check**           | `uvx ty check`                                                            |
| **Upgrade a dependency** | `uv add --dev <pkg>@latest && uv lock --upgrade-package <pkg> && uv sync` |
| **Generate lockfile**    | `uv lock && uv sync`                                                      |

> Use `uv run --` instead of `uvx` for tools that need the project installed (e.g. pytest) – see uv docs.

---

## 1  Introduction

**live‑crew** is a thin orchestration layer that enables standard CrewAI crews to run concurrently over real‑time event streams. Teams write normal CrewAI agents/tasks/crews and orchestrate them through YAML configuration, gaining deterministic timing, global shared context, replay capabilities, and NATS transport without modifying their CrewAI code.

### 1.1  Why live‑crew

- **Causal order** – hard 500 ms (configurable) beats called *TimeSlices*.
- **Concurrent crews** – each CrewAI package runs in its own runner; dependencies guarantee the right order.
- **GlobalContext** – a key‑value snapshot auto‑published each slice.
- **Transport & durability** – NATS JetStream for events/actions + KV.
- **Ops‑friendly** – heartbeats, Prometheus metrics, hot reload, replay.

### 1.2  TL;DR Features

```text
✔ Generic Event/Action models        ✔ Declarative dependencies
✔ Slice scheduler (offset maths)     ✔ Shared context diff‑merge
✔ Standard CrewAI agents/tasks/crews ✔ Thin wrapper over CrewAI
✔ YAML-driven orchestration          ✔ Python-defined CrewAI support
✔ Both YAML & Python CrewAI patterns ✔ Unified protocol architecture
✔ Sub‑process & multi‑host runners   ✔ Vector‑store hooks (optional)
✔ Replay harness & CI schema checks  ✔ Zero CrewAI modifications required
```


### 1.3  Pillars at a Glance

| Pillar                       | What it Means                                                                                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Time‑slicing**             | The world freezes every **500 ms** (configurable) into an integer **TimeSlice**; all ordering & replay anchor to this clock.                                 |
| **GlobalContext**            | Durable KV snapshot shared by all crews – `ctx[...]` assignment diff‑merged and re‑broadcast next slice.                                                     |
| **Declarative Dependencies** | Crews wake on `triggers` but may *wait* for other crews or event‑kinds with **offsets** (`crew: vision, offset:-1`).                                         |
| **Scheduler**                | Async orchestrator that assigns slices, publishes `slice.<n>.<crew>.ready`, commits context, emits heart‑beats, disables crash‑looping crews after 3 errors. |
| **Transport layer**          | NATS JetStream subjects for Events & Actions, JetStream KV for context, plus pub‑sub control subjects.                                                       |
| **Developer ergonomics**     | 90 % of a crew can be YAML‑only; drop to Python `react()` only when you need logic.                                                                          |
| **Ops features**             | Prometheus metrics, replay harness, hot reload, subprocess isolation, docker‑compose demo, MIT licence.                                                      |

---

## 2  Core Concepts & Reference Code

### 2.0  Epoch0 Rule

For each `stream_id`, **`epoch0` is the UTC timestamp of the first `Event` received**.
All slice indices are calculated relative to `epoch0`, guaranteeing deterministic replay across restarts.

### 2.1  Event & Action

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar, Any

PayloadT = TypeVar('PayloadT')

@dataclass(frozen=True, slots=True)
class Event(Generic[PayloadT]):
    ts: datetime
    kind: str
    stream_id: str
    payload: PayloadT

@dataclass(frozen=True, slots=True)
class Action(Generic[PayloadT]):
    ts: datetime
    kind: str
    stream_id: str
    payload: PayloadT
    ttl_ms: int = 5_000
````

### 2.2  TimeSlice Helper

```python
SLICE_MS = 500  # configurable

def slice_index(ts: datetime, epoch0: datetime) -> int:
    """Return the zero‑based slice index for timestamp *ts*."""
    return int(((ts - epoch0).total_seconds()*1000)//SLICE_MS)
```

### 2.3  GlobalContext & ContextProxy

The scheduler maintains a single **GlobalContext** that every crew can read and update. Mutations are recorded per‑slice and merged shallowly – *last‑writer‑wins*.

- Size cap: **4 KB**. If a snapshot exceeds this, the scheduler publishes `__error` with `ctx_size` overflow and truncates keys beyond the limit.
- Back‑ends: default JetStream KV (`live.<stream>.ctx`). Alternative implementations **MAY** use in‑memory or Redis – the interface must stay compatible.

```python
from collections.abc import MutableMapping
from typing import Any, Iterator

class ContextProxy(MutableMapping[str, Any]):
    """Dict‑like view that records incremental diffs."""
    ...
```

### 2.4  Dependency (discriminated union)

```python
from typing import Annotated, Union, Literal
from pydantic import BaseModel, Field

class CrewDep(BaseModel):
    type: Literal['crew']
    crew: str
    offset: int = -1

class EventDep(BaseModel):
    type: Literal['event']
    event: str
    offset: int = 0

Dependency = Annotated[Union[CrewDep, EventDep], Field(discriminator='type')]
```

### 2.5  Scheduler Pseudo‑Code (abridged)

```python
class Scheduler:
    """Slice scheduler – simplified pseudo‑code."""
    ...
```

*Reserved subject constants*

```
__context   live.<stream>.ctx snapshot
__hb        heartbeat event
__error     scheduler/crew error sink
```

### 2.6  CrewRunner & Helper APIs

```python
class EventBuffer: ...
class CrewRunner: ...
```

### 2.7  End‑to‑End Flow Example (Match‑day)

1. **Gateway** pushes `frame_embedding` & `goal_scored` Events into `live.match42.events`.
2. Scheduler bins them into **slice 83** (epoch0 + 41.5 s).
3. Publishes `__context` snapshot for slice 84.
4. Resolves dependencies:
   - `score_tracker` ready (needs `goal_scored@0`).
   - `commentator` waits (needs `vision_crew@‑1`).
   - `emoji_overlay` ready (`any` policy).
5. Publishes ready subjects.
6. `react()` runs → `score_tracker` bumps context & emits `score_update`; `emoji_overlay` emits confetti.
7. Context diff stored; appears next slice.
8. Slice 84 opens → `commentator` now ready → emits hype commentary line.
9. Graphics adapter updates score bug (< **5 ms** goal‑to‑graphic). TTS speaks line (\~ **0.5 s** goal‑to‑speech).
10. Loop continues with slice 85 …

---

## 3  YAML Artefacts

### 3.1  Vanilla CrewAI files

Unmodified `agents.yaml`, `tasks.yaml`, `crew.yaml` (CrewAI ≥ 1.0), etc.

### 3.2  `<crew>.runtime.yaml` Example

```yaml
crew: "commentary"
triggers: ["goal_scored"]
needs:
  - type: crew
    crew: "vision"
    offset: -1
wait_policy: any   # any | all | none | timeout:<ms>
slice_stride: 1
timeout_ms: 100
```

### 3.3  `live-config.yaml` (optional overrides)

```yaml
slice_ms: 250
heartbeat_s: 3
kv_backend: redis      # jetstream | redis | memory
vector:
  backend: "qdrant"
  url: "http://qdrant:6333"
```

### 3.4  YAML‑Only Auto‑Runner

A crew can be **pure YAML**:

```yaml
system_prompt: "You are an excitable commentator."
user_template: "{{ ctx['score']['home'] }}‑{{ ctx['score']['away'] }} at {{ slice }}"
patch_template: |
  ctx['last_line'] = outputs[0]['text']
outputs:
  - kind: commentary_line
    text: "{{ generated }}"
```

The scheduler executes a one‑shot CrewAI call using `system_prompt` + `user_template`, then optionally applies a Jinja `patch_template` to mutate context – no Python required.

---

## 4  Runtime Architecture

```
live.<stream>.events                  # raw input
live.<stream>.actions                 # output
live.<stream>.slice.<n>.<crew>.ready
live.<stream>.slice.<n>.<crew>.done
live.<stream>.errors                  # __error
live.<stream>.heartbeats              # __hb
live.<stream>.ctx                     # GlobalContext KV bucket
```

*Alert rule:* heartbeat gap > 15 s ⇒ PagerDuty.

Crews disabled after 3 consecutive errors (`reason:"disabled"`).

---

## 5  Developer Workflow

1. Write CrewAI package.
2. Add `<crew>.runtime.yaml`.
3. (Optional) Custom Python runner.
4. `live‑crew run -s match42`.
5. Replay harness for CI.
6. Hot‑reload after YAML change.
7. Distributed: `LIVE_CREW_SELECTOR='(commentator|ad_overlay)' live-crew run`.

Acceptance criteria for every PR must be embedded verbatim in the prompt.

## 5.1  CrewAI Integration Philosophy

**Thin Wrapper Approach**: live-crew serves as a minimal orchestration layer around standard CrewAI crews rather than replacing CrewAI functionality.

### Key Integration Principles

| Principle                    | Implementation                                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Zero CrewAI Modifications** | Standard CrewAI files (`agents.yaml`, `tasks.yaml`, `crew.yaml`) remain completely unchanged                                                               |
| **YAML-First Configuration** | 90% orchestration through YAML config files, 10% Python glue code for loading and execution                                                               |
| **Standard CrewAI Patterns** | Context accessed through normal CrewAI memory patterns, no live-crew-specific APIs required                                                                |
| **Framework Portability**    | CrewAI crews remain fully portable and executable outside live-crew environment                                                                            |
| **Progressive Enhancement**   | Add live-crew orchestration to existing CrewAI projects without code changes                                                                               |

### Minimal Integration Example

```python
# Complete multi-crew orchestration in <10 lines
from live_crew import CrewOrchestrator

# Standard CrewAI crews in: crews/analytics/, crews/reporting/
# Each contains: agents.yaml, tasks.yaml, crew.yaml, <crew>.runtime.yaml

orchestrator = CrewOrchestrator.from_config("live_crew_config.yaml")
await orchestrator.run("events.json")  # Routes events, manages dependencies, converts outputs
```

### Alternative: Python-Defined CrewAI Approach

```python
# Python-defined CrewAI crews with live-crew orchestration
from live_crew import CrewOrchestrator
from crewai import Agent, Task, Crew

# Standard CrewAI Python definitions
analytics_agent = Agent(role="Data Analyst", goal="Analyze user patterns")
analytics_task = Task(description="Analyze signup data", agent=analytics_agent)
analytics_crew = Crew(agents=[analytics_agent], tasks=[analytics_task])

orchestrator = CrewOrchestrator()
orchestrator.register_crew("analytics", analytics_crew,
                          triggers=["user_signup"], timeout_ms=200)
await orchestrator.run("events.json")
```

### Complete Integration Example

**Project Structure** (standard CrewAI + live-crew orchestration):
```
project/
├── live_crew_config.yaml          # Master orchestration config
├── events.json                     # Input events
├── crews/
│   ├── user_analytics/
│   │   ├── agents.yaml             # Standard CrewAI agents
│   │   ├── tasks.yaml              # Standard CrewAI tasks
│   │   ├── crew.yaml               # Standard CrewAI crew
│   │   └── user_analytics.runtime.yaml  # live-crew orchestration
│   └── reporting/
│       ├── agents.yaml             # Standard CrewAI agents
│       ├── tasks.yaml              # Standard CrewAI tasks
│       ├── crew.yaml               # Standard CrewAI crew
│       └── reporting.runtime.yaml      # live-crew orchestration
└── main.py                         # 8-line orchestration script
```

**Configuration Files**:
```yaml
# live_crew_config.yaml - Master orchestration
crews:
  - path: "crews/user_analytics"
    runtime: "user_analytics.runtime.yaml"
  - path: "crews/reporting"
    runtime: "reporting.runtime.yaml"

# user_analytics.runtime.yaml - Independent crew
crew: "user_analytics"
triggers: ["user_signup", "user_activity"]
wait_policy: none
timeout_ms: 200

# reporting.runtime.yaml - Depends on analytics
crew: "reporting"
triggers: ["user_signup"]
needs:
  - type: crew
    crew: "user_analytics"
    offset: -1
wait_policy: all
timeout_ms: 500
```

**Python Orchestration Script** (complete implementation):
```python
# main.py - 8 lines total for multi-crew orchestration
import asyncio
from live_crew import CrewOrchestrator

async def main():
    orchestrator = CrewOrchestrator.from_config("live_crew_config.yaml")
    await orchestrator.run("events.json")

if __name__ == "__main__":
    asyncio.run(main())
```

The standard CrewAI crews remain completely unchanged and portable. live-crew handles event routing, dependency coordination, context sharing, and output conversion automatically.

## 5.2  Migration from Decorator Pattern to YAML-Driven Approach

**Current State vs Target Architecture**:

| Aspect                   | Current Decorator Approach                | Target YAML-Driven Approach                    |
| ------------------------ | ----------------------------------------- | ----------------------------------------------- |
| **Event Handling**      | `@event_handler("user_signup")`          | YAML `triggers: ["user_signup"]`               |
| **Crew Logic**          | Python functions with custom logic       | Standard CrewAI crews with agents/tasks        |
| **Dependencies**        | Manual coordination in Python code       | YAML `needs:` declarations with offsets        |
| **Context Access**      | Custom context parameter injection        | Standard CrewAI memory patterns                 |
| **Learning Curve**      | Framework-specific patterns required      | Extend existing CrewAI knowledge               |
| **Code Portability**    | Tied to live-crew framework             | Standard CrewAI crews remain portable          |

**Implementation Strategy**:
1. **Phase 1**: Maintain backward compatibility for existing decorator-based examples
2. **Phase 2**: Implement CrewAI crew loader and YAML orchestration system
3. **Phase 3**: Provide migration tools to convert decorator handlers to CrewAI crews
4. **Phase 4**: Deprecate decorator approach in favor of YAML-driven CrewAI integration

**Benefits of YAML-Driven Approach**:
- **Leverage CrewAI Ecosystem**: Users benefit from CrewAI's agent/task abstractions and community
- **Reduce Learning Curve**: Extend CrewAI knowledge instead of learning new framework patterns
- **Improve Portability**: CrewAI crews work independently of live-crew for testing and development
- **Enhance Maintainability**: Configuration-driven approach reduces code complexity
- **Enable Scalability**: YAML configuration supports complex multi-crew orchestration patterns

---

## 6  Performance & Reliability Knobs

| Knob            | Default | Accepted Values                      | Purpose                          |
| --------------- | ------- | ------------------------------------ | -------------------------------- |
| slice\_stride   | 1       | N ≥ 1                                | Run 1/N slices                   |
| timeout\_ms     | ∞       | positive int                         | Hard wall‑clock kill per `react` |
| backlog\_policy | fifo    | `fifo` \| `latest`                   | Backlog collapse strategy        |
| wait\_policy    | —       | `all`\|`any`\|`none`\|`timeout:<ms>` | Dependency satisfaction rule     |
| max\_errors     | 3       | ≥ 1                                  | Disable crew after N errors      |
| Heartbeat       | 5 s     | ≥ 1 s                                | Liveness watchdog                |

---

## 7  GlobalContext Usage Example

```python
ctx['score']['home'] = 1
ctx['score']['away'] = 0
```

---

## 8  Extensibility Hooks & Adapter Payloads

Crews call built‑in tools via `tools:` list – ``** and **``.

```jsonc
// Vector search adapter – returns list[Document]
{
  "kind": "vector_search",
  "model": "text-embedding-3-small",
  "query": "fast counter-attack highlights",
  "top_k": 8
}

// Fact insert adapter – muted commentary overlay
{
  "kind": "fact_insert",
  "fact": "Team X has won 5 consecutive matches",
  "slice": 123
}
```

### Adapter payload reference

```jsonc
// score_update Action
{
  "kind": "score_update",
  "stream_id": "match42",
  "ts": "2025-07-31T19:22:05.200Z",
  "payload": { "home": 2, "away": 1 }
}

// commentary_line Action
{
  "kind": "commentary_line",
  "stream_id": "match42",
  "ts": "2025-07-31T19:22:06.900Z",
  "payload": {
    "text": "GOAL! 2–1 Madrid.",
    "language": "en-US"
  }
}
```

---

---

## 9  Deployment & Ops

```yaml
version: '3'
services:
  nats:
    image: nats:2.10
    command: ["-js"]
    ports: ["4222:4222"]
  live-crew:
    build: .
    depends_on: [nats]
    environment:
      - STREAM=match42
      - LOG_LEVEL=INFO
```

### 9.1  Prometheus Metrics

| Metric                     | Type    | Description              |
| -------------------------- | ------- | ------------------------ |
| `slice_lag_seconds{crew}`  | gauge   | Scheduler → crew latency |
| `actions_sent_total{kind}` | counter | Actions emitted          |
| `errors_total{crew}`       | counter | Exceptions bubbled up    |
| `heartbeats_total`         | counter | Heartbeats published     |

### 9.2  Distributed Runners

Run a subset of crews on another host:

```bash
LIVE_CREW_SELECTOR='(commentator|ad_overlay)' live-crew run -s match42
```

Subjects are plain pub‑sub, so multiple runners share the same NATS cluster.

### 9.3  Security & Compliance

- Snapshot cap 4 KB mitigates log‑flood DoS.
- Honour NATS **ACLs** – each crew can be sandboxed to its own JetStream/KV bucket.
- MIT licence; no GPL dependencies.
- External API keys pulled from env vars → easy stubbing.

---

## 10  Logging

Components MUST emit structured JSON:

```jsonc
{
  "ts": "2025-07-31T12:00:00Z",
  "level": "INFO",
  "msg": "slice complete",
  "stream": "match42",
  "slice": 42,
  "crew": "vision",
  "t_ms": 27
}
```

---

## 11  Licensing & Naming

MIT licence; not affiliated with CrewAI.

---

## 12  Roadmap & Milestones

| #  | Milestone (Done when …)                     |
| -- | ------------------------------------------- |
| 1  | Core message model & minimal scheduler      |
| 2  | Multi‑crew concurrency & Context diff‑merge |
| 3  | Performance knobs & error sink              |
| 4  | Sub‑process runners & Prometheus            |
| 5  | Vector clip finder adapter                  |
| 6  | CLI scaffolding & validate                  |
| 7  | Structured logging                          |
| 8  | Heartbeat alerts                            |
| 9  | Replay harness v2                           |
| 10 | Alternative Context back‑ends               |
| 11 | Vector store adapters                       |
| 12 | Security hardening                          |
| 13 | Production release                          |

---

## 13  Appendices

### 13.1  CLI Reference (Typer)

```python
import typer
from pathlib import Path

app = typer.Typer(help="live‑crew orchestration CLI")

@app

```
