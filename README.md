# live-crew

**A low-latency, slice-based orchestration layer for running multiple CrewAI crews concurrently over real-time event streams.**

Orchestrate AI crews with deterministic timing, shared context, and event-driven coordination. Perfect for real-time applications where multiple AI agents need to work together within strict timing constraints.

---

## 🚀 For Framework Users

*Use live-crew to orchestrate CrewAI crews in your applications.*

### What is live-crew?

live-crew enables you to run multiple CrewAI crews simultaneously with:

- **Deterministic timing**: Process events in configurable time slices (500ms by default)
- **Shared context**: Crews can share data and coordinate through a global key-value store
- **Event-driven**: React to real-time events from files, NATS, or custom transports
- **Dependency management**: Control execution order between crews
- **Replay capabilities**: Deterministic replay for testing and debugging

Perfect for applications like real-time dashboards, live sports commentary, financial trading systems, or any scenario where multiple AI agents need to coordinate in real-time.

### Quick Start

#### Installation

**Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/)**

```bash
# Install from PyPI (when published)
uv add live-crew

# Or clone and install locally
git clone https://github.com/your-username/live-crew.git
cd live-crew
uv sync
```

#### Hello World Example

Here's a complete working example you can copy-paste and run:

```python
#!/usr/bin/env python3
"""Hello World example for live-crew.

This example shows how to:
1. Create and publish events
2. Set up basic configuration
3. Use time slicing for deterministic processing
"""

import asyncio
from datetime import datetime, timezone
from live_crew.core.models import Event, Action
from live_crew.config.settings import load_config
from live_crew.core.timeslice import slice_index

async def hello_world():
    """Basic live-crew usage demonstration."""

    # Load configuration (uses defaults if no config file)
    config = load_config()
    print(f"⚙️  Config: {config.slice_ms}ms slices, {config.kv_backend} backend")

    # Create an event (immutable, validated)
    event = Event[dict](
        ts=datetime.now(timezone.utc),
        kind="user_message",
        stream_id="chat_001",
        payload={"message": "Hello, live-crew!", "user": "alice"}
    )
    print(f"📥 Event: {event.kind} from {event.stream_id}")

    # Create a response action with TTL
    action = Action[str](
        ts=datetime.now(timezone.utc),
        kind="ai_response",
        stream_id="chat_001",
        payload="Hello! I'm processing your message in real-time.",
        ttl_ms=10000  # expires after 10 seconds
    )
    print(f"📤 Action: {action.kind} (expires in {action.ttl_ms}ms)")

    # Demonstrate time slicing
    epoch0 = datetime.now(timezone.utc)
    event_slice = slice_index(event.ts, epoch0, config.slice_ms)
    action_slice = slice_index(action.ts, epoch0, config.slice_ms)

    print(f"⏰ Time slicing:")
    print(f"   Event in slice: {event_slice}")
    print(f"   Action in slice: {action_slice}")
    print(f"   Slice duration: {config.slice_ms}ms")

if __name__ == "__main__":
    asyncio.run(hello_world())
```

Save this as `hello.py` and run:

```bash
python hello.py
```

Expected output:
```
⚙️  Config: 500ms slices, jetstream backend
📥 Event: user_message from chat_001
📤 Action: ai_response (expires in 10000ms)
⏰ Time slicing:
   Event in slice: 0
   Action in slice: 0
   Slice duration: 500ms
```

#### Configuration

**Option 1: YAML Configuration** (`live-config.yaml`)

```yaml
# Time slicing configuration
slice_ms: 500        # Process events every 500ms (1-10000ms range)
heartbeat_s: 30      # Heartbeat interval for health checks

# Storage backends
kv_backend: jetstream # jetstream | redis | memory

# Optional vector store for embeddings
vector:
  backend: qdrant
  url: http://localhost:6333
```

**Option 2: Environment Variables** (override YAML)

```bash
export LIVE_CREW_SLICE_MS=250      # Faster processing (250ms slices)
export LIVE_CREW_HEARTBEAT_S=10    # More frequent heartbeats
export LIVE_CREW_KV_BACKEND=memory # Use in-memory storage for development
```

#### Working with Events and Actions

Events and actions are the core data types in live-crew:

```python
from datetime import datetime, timezone
from live_crew.core.models import Event, Action

# Events: Immutable input data
event = Event[dict](
    ts=datetime.now(timezone.utc),
    kind="stock_price_update",      # alphanumeric + underscore only
    stream_id="nasdaq_aapl",        # alphanumeric + underscore/dash
    payload={"symbol": "AAPL", "price": 150.25, "volume": 1000}
)

# Actions: Immutable output data with TTL
action = Action[str](
    ts=datetime.now(timezone.utc),
    kind="trading_signal",
    stream_id="nasdaq_aapl",
    payload="BUY signal for AAPL at $150.25",
    ttl_ms=30000  # Valid for 30 seconds
)

# Type safety with generics
user_event = Event[dict](
    ts=datetime.now(timezone.utc),
    kind="user_login",
    stream_id="session_123",
    payload={"user_id": "alice", "ip": "192.168.1.1"}
)

notification = Action[dict](
    ts=datetime.now(timezone.utc),
    kind="push_notification",
    stream_id="session_123",
    payload={"title": "Welcome back!", "body": "You have 3 new messages"},
    ttl_ms=60000
)
```

#### Understanding Time Slicing

Time slicing ensures deterministic, predictable processing:

```python
from datetime import datetime, timezone, timedelta
from live_crew.core.timeslice import slice_index

# Define processing epoch (start time)
epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
slice_duration = 500  # 500ms slices

# Events at different times fall into different slices
events = [
    datetime(2024, 1, 1, 12, 0, 0, 100000, tzinfo=timezone.utc),  # 100ms after epoch
    datetime(2024, 1, 1, 12, 0, 0, 600000, tzinfo=timezone.utc),  # 600ms after epoch
    datetime(2024, 1, 1, 12, 0, 1, 200000, tzinfo=timezone.utc),  # 1.2s after epoch
]

for i, event_time in enumerate(events):
    slice_idx = slice_index(event_time, epoch0, slice_duration)
    print(f"Event {i+1}: slice {slice_idx}")

# Output:
# Event 1: slice 0    (0-499ms)
# Event 2: slice 1    (500-999ms)
# Event 3: slice 2    (1000-1499ms)
```

This ensures that:
- Events are processed in predictable batches
- Replay produces identical results
- System behavior is deterministic across runs

#### Common Patterns

**Pattern 1: Event Processing with Context**

```python
# Process events and maintain context across slices
from live_crew.config.settings import load_config

config = load_config()
context = {}  # Shared context between crews

# Event processing would happen here with crews
# (Full crew integration examples coming in future releases)
```

**Pattern 2: Multi-Stream Processing**

```python
# Handle multiple data streams simultaneously
streams = ["market_data", "news_feed", "social_media"]

for stream in streams:
    event = Event[dict](
        ts=datetime.now(timezone.utc),
        kind="data_update",
        stream_id=stream,
        payload={"source": stream, "data": "..."}
    )
    # Process each stream independently
```

**Pattern 3: TTL-based Action Management**

```python
# Actions with different lifespans
immediate_action = Action[str](
    ts=datetime.now(timezone.utc),
    kind="alert",
    stream_id="system",
    payload="Critical error detected!",
    ttl_ms=1000  # Urgent, expires quickly
)

persistent_action = Action[dict](
    ts=datetime.now(timezone.utc),
    kind="analytics_update",
    stream_id="dashboard",
    payload={"metric": "cpu_usage", "value": 85.2},
    ttl_ms=300000  # Keep for 5 minutes
)
```

#### Next Steps

1. **Explore transports**: Learn about file, console, and NATS transports
2. **Set up crews**: Integrate CrewAI crews with live-crew orchestration
3. **Configure backends**: Choose between memory, Redis, or NATS JetStream storage
4. **Build applications**: Create real-time AI applications with coordinated crews

For advanced usage and CrewAI integration examples, see the Framework Developers section below.

---

## 🔧 For Framework Developers

*Understand live-crew's architecture and extend it with new implementations.*

### Architecture Overview

live-crew is built around a clean, protocol-based architecture that enables seamless progression from simple in-memory implementations to distributed, production-ready systems.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Events        │    │   Time Slices   │    │   Actions       │
│  (Input Data)   │────▶│  (Processing)   │────▶│ (Output Data)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ EventTransport  │    │ SchedulerBackend│    │ ActionTransport │
│   Protocol      │    │    Protocol     │    │    Protocol     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ContextBackend  │    │ CrewRegistry    │    │ EventHandler    │
│   Protocol      │    │   Protocol      │    │   Protocol      │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Data Flow:
Events → Scheduler → Dependencies → Crews → Context Updates → Actions
```

### Core Abstractions

All live-crew components are built around protocol interfaces, enabling easy extension and testing:

#### Transport Layer Protocols

**EventTransport Protocol** - Handle event ingestion
```python
from live_crew.interfaces import EventTransport
from live_crew.core.models import Event
from typing import Any, AsyncIterator

class CustomEventTransport:
    async def publish_event(self, event: Event[Any]) -> None:
        """Publish an event to your transport layer."""
        # Implementation: NATS, Kafka, HTTP webhooks, etc.
        pass

    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        """Subscribe to events from your transport layer."""
        # Implementation: yield events as they arrive
        while True:
            yield event  # Your event source here
```

**ActionTransport Protocol** - Handle action output
```python
from live_crew.interfaces import ActionTransport
from live_crew.core.models import Action

class CustomActionTransport:
    async def publish_action(self, action: Action[Any]) -> None:
        """Publish an action to your output system."""
        # Implementation: webhooks, databases, message queues
        pass

    async def subscribe_actions(self) -> AsyncIterator[Action[Any]]:
        """Subscribe to actions for monitoring/logging."""
        # Implementation: consume actions for observability
        pass
```

#### Storage Backend Protocols

**ContextBackend Protocol** - Manage shared state
```python
from live_crew.interfaces import ContextBackend
from typing import Any

class CustomContextBackend:
    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        """Get context snapshot for a specific stream and slice."""
        # Implementation: Redis, PostgreSQL, NATS KV, etc.
        return {}

    async def apply_diff(
        self, stream_id: str, slice_idx: int, diff: dict[str, Any]
    ) -> None:
        """Apply a context diff for a specific stream and slice."""
        # Implementation: atomic updates with diff-merge logic
        pass

    async def clear_stream(self, stream_id: str) -> None:
        """Clear all context data for a stream."""
        pass
```

**SchedulerBackend Protocol** - Manage crew execution
```python
from live_crew.interfaces import SchedulerBackend
from live_crew.core.dependencies import Dependency

class CustomSchedulerBackend:
    async def schedule_crew(
        self, crew_id: str, slice_idx: int, dependencies: list[Dependency]
    ) -> None:
        """Schedule a crew for execution in a specific slice."""
        # Implementation: dependency resolution, execution ordering
        pass

    async def mark_crew_complete(self, crew_id: str, slice_idx: int) -> None:
        """Mark a crew as completed for a specific slice."""
        pass

    async def get_pending_crews(self, slice_idx: int) -> list[str]:
        """Get list of crews pending execution for a slice."""
        return []
```

#### Crew System Protocols

**EventHandler Protocol** - Define crew behavior
```python
from live_crew.interfaces import EventHandler
from live_crew.core.models import Event, Action
from typing import Any

class CustomEventHandler:
    @property
    def crew_id(self) -> str:
        return "my_custom_crew"

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle an event and produce actions."""
        # Implementation: CrewAI integration, custom logic
        actions = []

        # Example: process event and create response
        if event.kind == "user_message":
            response = Action[str](
                ts=event.ts,
                kind="ai_response",
                stream_id=event.stream_id,
                payload=f"Processed: {event.payload}",
                ttl_ms=5000
            )
            actions.append(response)

        return actions
```

**CrewRegistry Protocol** - Manage crew lifecycle
```python
from live_crew.interfaces import CrewRegistry, EventHandler
from live_crew.core.dependencies import Dependency

class CustomCrewRegistry:
    def register_crew(
        self, handler: EventHandler, dependencies: list[Dependency]
    ) -> None:
        """Register a crew with its handler and dependencies."""
        pass

    def get_handler(self, crew_id: str) -> EventHandler | None:
        """Get the event handler for a crew."""
        return None

    def get_dependencies(self, crew_id: str) -> list[Dependency]:
        """Get the dependencies for a crew."""
        return []

    def list_crews(self) -> list[str]:
        """List all registered crew IDs."""
        return []
```

### Time Slicing Deep Dive

Time slicing is the heart of live-crew's deterministic processing:

#### How Time Slicing Works

```python
from live_crew.core.timeslice import slice_index
from datetime import datetime, timezone

def understand_time_slicing():
    """Demonstrate time slicing mechanics."""

    # Define epoch (processing start time)
    epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    slice_ms = 500  # 500ms slices

    # Events at different timestamps
    timestamps = [
        datetime(2024, 1, 1, 12, 0, 0, 100000, tzinfo=timezone.utc),  # +100ms
        datetime(2024, 1, 1, 12, 0, 0, 450000, tzinfo=timezone.utc),  # +450ms
        datetime(2024, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc),  # +500ms
        datetime(2024, 1, 1, 12, 0, 0, 999000, tzinfo=timezone.utc),  # +999ms
        datetime(2024, 1, 1, 12, 0, 1, 0, tzinfo=timezone.utc),       # +1000ms
    ]

    for ts in timestamps:
        slice_idx = slice_index(ts, epoch0, slice_ms)
        delta_ms = (ts - epoch0).total_seconds() * 1000
        print(f"Δ{delta_ms:4.0f}ms → slice {slice_idx}")

    # Output:
    # Δ 100ms → slice 0
    # Δ 450ms → slice 0
    # Δ 500ms → slice 1
    # Δ 999ms → slice 1
    # Δ1000ms → slice 2
```

#### Slice Index Calculation

The `slice_index` function provides deterministic time slicing:

```python
def slice_index(ts: datetime, epoch0: datetime, slice_ms: int = 500) -> int:
    """Return the zero-based slice index for timestamp ts."""
    delta_ms = (ts - epoch0).total_seconds() * 1000
    return int(delta_ms // slice_ms)
```

Key properties:
- **Deterministic**: Same inputs always produce same outputs
- **Integer division**: Ensures consistent binning
- **Zero-based**: First slice is index 0
- **Negative support**: Events before epoch0 get negative indices
- **Timezone agnostic**: Works with any timezone (but be consistent!)

#### Configurable Slice Duration

```python
# Different slice durations for different use cases
fast_slicing = 100   # 100ms - High-frequency trading
standard = 500       # 500ms - Real-time dashboards
batch = 5000        # 5s - Analytics processing

# Configuration via YAML or environment
config = load_config()
slice_ms = config.slice_ms  # Configurable slice duration
```

### Module Structure

live-crew follows a clean, modular architecture:

```
src/live_crew/
├── core/                    # Core data models and utilities
│   ├── models.py           # Event and Action Pydantic models
│   ├── timeslice.py        # Time slicing utilities
│   └── dependencies.py     # Dependency resolution system
│
├── interfaces/             # Protocol definitions
│   └── protocols.py        # All protocol interfaces
│
├── config/                 # Configuration management
│   └── settings.py         # Pydantic settings with YAML support
│
├── transports/             # Transport implementations
│   ├── file.py            # File-based transport (Slice 1)
│   ├── console.py         # Console output transport
│   └── nats.py            # NATS transport (coming in Slice 2)
│
├── backends/               # Storage backend implementations
│   ├── context.py         # Context storage backends
│   └── vector.py          # Vector storage backends (planned)
│
├── scheduling/             # Scheduler implementations
│   ├── memory.py          # In-memory scheduler (Slice 1)
│   └── distributed.py     # Distributed scheduler (Slice 2, planned)
│
└── crew/                   # Crew management
    ├── definition.py       # Crew definition models
    ├── handlers.py         # Event handler implementations
    └── registry.py         # Crew registry implementations
```

### Extension Points

#### Adding New Transports

1. **Implement the protocols**:
```python
from live_crew.interfaces import EventTransport, ActionTransport

class NATSTransport:
    """NATS-based transport implementation."""

    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        # Initialize NATS connection

    async def publish_event(self, event: Event[Any]) -> None:
        # Publish to NATS JetStream
        pass

    async def subscribe_events(self) -> AsyncIterator[Event[Any]]:
        # Subscribe to NATS stream
        pass
```

2. **Register in configuration**:
```python
# Add to config/settings.py
transport_backend: Literal["file", "console", "nats"] = "nats"
```

3. **Use in applications**:
```python
transport = NATSTransport("nats://prod-cluster:4222")
# Integrate with live-crew orchestrator
```

#### Adding New Storage Backends

1. **Implement ContextBackend**:
```python
from live_crew.interfaces import ContextBackend
import redis.asyncio as redis

class RedisContextBackend:
    """Redis-based context storage with diff-merge support."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)

    async def get_snapshot(self, stream_id: str, slice_idx: int) -> dict[str, Any]:
        key = f"context:{stream_id}:{slice_idx}"
        data = await self.redis.hgetall(key)
        return {k.decode(): json.loads(v) for k, v in data.items()}

    async def apply_diff(
        self, stream_id: str, slice_idx: int, diff: dict[str, Any]
    ) -> None:
        # Implement atomic diff application
        pipe = self.redis.pipeline()
        key = f"context:{stream_id}:{slice_idx}"
        for field, value in diff.items():
            pipe.hset(key, field, json.dumps(value))
        await pipe.execute()
```

2. **Add configuration support**:
```yaml
# live-config.yaml
kv_backend: redis
redis:
  url: redis://localhost:6379
  pool_size: 10
```

#### Adding New Schedulers

```python
from live_crew.interfaces import SchedulerBackend
from live_crew.core.dependencies import Dependency

class DistributedScheduler:
    """Distributed scheduler using NATS for coordination."""

    async def schedule_crew(
        self, crew_id: str, slice_idx: int, dependencies: list[Dependency]
    ) -> None:
        # Implement distributed scheduling logic
        # - Check dependencies across cluster
        # - Coordinate execution with other nodes
        # - Handle failover and recovery
        pass
```

### Development Setup

#### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

#### Development Installation

```bash
# Clone repository
git clone https://github.com/your-username/live-crew.git
cd live-crew

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run pytest
```

#### Code Quality Tools

```bash
# Linting and formatting
uv run ruff check --fix    # Auto-fix linting issues
uv run ruff format         # Format code

# Type checking
uv run mypy src/live_crew  # Static type analysis

# Testing
uv run pytest             # Run all tests
uv run pytest -v          # Verbose output
uv run pytest --cov       # Coverage report

# Pre-commit hooks (run automatically on commit)
uv run pre-commit run --all-files
```

#### Testing Strategy

live-crew uses comprehensive testing:

```bash
# Test categories
uv run pytest tests/test_models.py           # Core data models
uv run pytest tests/test_timeslice.py        # Time slicing logic
uv run pytest tests/test_config.py           # Configuration loading
uv run pytest tests/test_dependencies.py     # Dependency resolution
uv run pytest tests/test_pattern_validation.py # Input validation

# Integration tests (when implemented)
uv run pytest tests/integration/             # End-to-end scenarios
```

#### Contributing Guidelines

1. **Follow the protocol-based architecture** - All new components should implement the appropriate protocols
2. **Maintain backward compatibility** - Changes should not break existing implementations
3. **Add comprehensive tests** - New features require unit and integration tests
4. **Update documentation** - Keep README and docstrings current
5. **Use type hints** - All code should be fully typed
6. **Follow code style** - Use ruff for formatting and linting

#### Project Management

This project uses a `.vibes` folder for documentation and project management:

```
.vibes/
├── live_crew_spec.md      # Complete technical specification
├── scrum.md              # Sprint planning and progress
└── references/           # Library documentation and examples
```

The `.vibes` folder is part of an experiment with [Claude Code](https://claude.ai/code) to improve AI-assisted development workflows.

---

## License

MIT License - see LICENSE file for details.
