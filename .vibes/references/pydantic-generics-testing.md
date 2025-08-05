# Pydantic Generics Testing Reference

## The Problem: Freezegun + Pydantic v2 + Generics

When using Freezegun with Pydantic v2 generic models that contain datetime fields, schema generation fails with:

```
PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'datetime.datetime'>
```

This occurs because:
1. Freezegun patches `datetime.datetime` at runtime
2. Pydantic v2's schema generation happens lazily when `Event[PayloadType]` is first called
3. The schema generator doesn't recognize the patched datetime as a standard datetime type

## Strategic Solutions (Ranked by Quality)

### 1. Schema Pre-Generation (RECOMMENDED)
Create generic model instances before Freezegun patches datetime:

```python
# At module level or in setup
from live_crew.core.models import Event
from typing import Any

# Pre-generate schemas before freezegun patches anything
EventDict = Event[dict[str, Any]]
EventAny = Event[Any]

# In tests, use the pre-generated types
@freeze_time("2025-07-20T12:00:00Z")
def test_event_creation():
    event = EventDict(**event_data)  # Works - schema already generated
```

### 2. Test Fixtures with Schema Loading
Use pytest fixtures to ensure schemas are loaded early:

```python
@pytest.fixture(scope="session", autouse=True)
def preload_pydantic_schemas():
    """Pre-load Pydantic schemas before any freezegun usage."""
    from live_crew.core.models import Event
    from typing import Any

    # Force schema generation
    _ = Event[dict[str, Any]]
    _ = Event[Any]
    _ = Event[dict]
```

### 3. Custom Test Event Types
Create concrete Event types for testing:

```python
# In test utilities
class TestEvent(Event[dict[str, Any]]):
    """Concrete Event type for testing."""
    pass

class TestEventAny(Event[Any]):
    """Any-payload Event type for testing."""
    pass
```

### 4. Alternative Time Mocking
Switch to `time-machine` which doesn't interfere with imports:

```bash
uv add --dev time-machine
```

```python
import time_machine

@time_machine.travel("2025-07-20T12:00:00Z")
def test_event_creation():
    event = Event[dict](**event_data)  # Works without schema issues
```

## Best Practices for Generic Model Testing

### Type Safety in Tests
Always use specific payload types rather than `Any`:

```python
# Good - specific types
EventDict = Event[dict[str, Any]]
EventStrPayload = Event[str]
EventUserPayload = Event[UserData]

# Avoid - loses type safety
EventAny = Event[Any]
```

### Schema Generation Timing
- Generate schemas early in test setup
- Use module-level pre-generation for commonly used types
- Consider session-scoped fixtures for expensive schema generation

### Payload Type Patterns
```python
# For simple data
Event[dict[str, Any]]

# For specific models
Event[UserCreatedPayload]

# For JSON-like data
Event[dict[str, str | int | float | bool | None]]

# For completely flexible testing only
Event[Any]
```

## Implementation Strategy

1. **Pre-generate common Event types** at module level
2. **Use specific payload types** instead of bare `dict` or `Any`
3. **Create test utilities** for commonly used Event patterns
4. **Consider time-machine** as a Freezegun alternative if issues persist
5. **Document the pattern** for team consistency

## Common Pitfalls

- **Late schema generation**: Don't create `Event[T]` inside frozen time contexts
- **Overusing Any**: Loses type safety benefits
- **arbitrary_types_allowed**: Nuclear option that disables important validation
- **Ignoring the root cause**: This is a timing issue, not a model design issue

## Performance Considerations

- Schema generation is expensive - do it once at module load
- Session-scoped fixtures avoid repeated schema generation
- Pre-generated types have no runtime overhead
