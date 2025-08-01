# live-crew

A low-latency, slice-based orchestration layer for running multiple CrewAI crews concurrently over real-time event streams.

## Overview

live-crew adds deterministic timing, global shared context, replay capabilities, and NATS transport to CrewAI while preserving its agent/task abstractions. It's designed for real-time applications where multiple AI crews need to coordinate and react to events within configurable timing constraints.

## Key Features

- **Time-slicing**: Deterministic beats for predictable event processing
- **Concurrent crews**: Multiple CrewAI packages running simultaneously with dependency management
- **Global context**: Shared key-value store with diff-merge updates
- **NATS transport**: JetStream for events/actions, KV for context persistence
- **Replay system**: Deterministic replay capabilities for testing and debugging
- **Performance knobs**: Configurable timeouts, error handling, and backlog policies

## Architecture

```
Events → Scheduler → TimeSlices → Dependency Resolution → Crew Execution → Actions
```

Events are binned into time slices, crews execute based on dependencies, and all share a global context that's updated each slice.

## Quick Start

### Installation

**Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/)**

```bash
# Clone and setup
git clone https://github.com/your-username/live-crew.git
cd live-crew

# Install dependencies and setup pre-commit hooks
uv sync --dev && uv run pre-commit install

# Verify installation
uv run pytest
```

### Basic Usage

```python
from live_crew.models import Event, Action
from live_crew.config import load_config
from live_crew.timeslice import slice_index
from datetime import datetime, timezone

# Configuration - loads from live-config.yaml or environment variables
config = load_config()
print(f"Time slice: {config.slice_ms}ms, backend: {config.kv_backend}")

# Create events (immutable, validated)
event = Event[dict](
    ts=datetime.now(timezone.utc),
    kind="goal_scored",           # alphanumeric + underscore only
    stream_id="match42",          # alphanumeric + underscore/dash
    payload={"team": "home", "player": "Messi"}
)

# Create actions with TTL (time-to-live)
action = Action[str](
    ts=datetime.now(timezone.utc),
    kind="commentary",
    stream_id="match42",
    payload="GOAL! What a strike!",
    ttl_ms=5000  # expires after 5 seconds
)

# Time slicing for deterministic processing
epoch0 = datetime.now(timezone.utc)
slice_idx = slice_index(event.ts, epoch0)
print(f"Event in slice: {slice_idx}")
```

### Configuration

**Option 1: YAML file** (`live-config.yaml`)
```yaml
slice_ms: 500        # 500ms time slices (1-10000ms)
heartbeat_s: 30      # 30s heartbeat interval (1-300s)
kv_backend: jetstream # jetstream | redis | memory
vector:              # Optional vector store
  backend: qdrant
  url: http://qdrant:6333
```

**Option 2: Environment variables** (override YAML)
```bash
export LIVE_CREW_SLICE_MS=250
export LIVE_CREW_HEARTBEAT_S=5
export LIVE_CREW_KV_BACKEND=redis
```

### Development

```bash
# Run tests (88 tests, comprehensive validation)
uv run pytest

# Code quality (pre-commit hooks auto-run on commit)
uv run ruff check --fix    # lint & auto-fix
uv run ruff format         # format code
uv run pre-commit run --all-files  # run all hooks

# Manual pre-commit hook run
uv run pre-commit run --all-files
```

## Project Structure

- `.vibes/` - Documentation and project management (for Claude Code experimentation)
  - `live_crew_spec.md` - Complete technical specification
  - `scrum.md` - Sprint planning and progress tracking
  - `references/` - Library documentation
- `CLAUDE.md` - Development guidelines for Claude Code

## About .vibes

This project uses a `.vibes` folder for documentation and project management as part of an experiment with [Claude Code](https://claude.ai/code). The folder contains detailed specifications, sprint planning, and reference materials to help AI assistants understand and contribute to the project effectively.

## License

MIT License - see LICENSE file for details.
