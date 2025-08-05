# FreezeGun Library Reference

## Overview

FreezeGun is a library that allows Python tests to "travel through time" by mocking the datetime module. It's essential for creating deterministic tests that involve time-dependent code.

**Key Benefits:**
- Eliminates non-deterministic test failures due to time dependencies
- Makes tests reproducible across different execution times
- Allows testing edge cases involving specific dates/times
- Supports timezone handling and time progression

## Installation

```bash
# Basic installation
uv add --dev freezegun

# With pytest plugin (optional but recommended)
uv add --dev pytest-freezegun
```

## Core Usage Patterns

### 1. Decorator Pattern (Most Common)

```python
from freezegun import freeze_time
from datetime import datetime

@freeze_time("2025-01-15T12:30:00Z")
def test_user_registration():
    """Test with frozen time."""
    user = create_user("john@example.com")
    assert user.created_at == datetime(2025, 1, 15, 12, 30, 0)

@freeze_time("2025-01-15")  # Date only
def test_daily_report():
    """Test with date-only freeze."""
    report = generate_daily_report()
    assert report.date == datetime(2025, 1, 15).date()
```

### 2. Context Manager Pattern

```python
def test_time_dependent_logic():
    """Test with context manager for precise control."""
    # Before freeze
    initial_time = datetime.now()

    with freeze_time("2025-01-15T10:00:00Z"):
        # Time is frozen
        frozen_time = datetime.now()
        assert frozen_time == datetime(2025, 1, 15, 10, 0, 0)

        # Do time-dependent operations
        result = process_events_at_time()
        assert result.timestamp == frozen_time

    # After context - time is unfrozen
    current_time = datetime.now()
    assert current_time != frozen_time
```

### 3. Pytest Fixture Pattern

```python
import pytest
from freezegun import freeze_time

@pytest.fixture
def frozen_time():
    """Fixture to freeze time for multiple tests."""
    with freeze_time("2025-01-15T12:00:00Z"):
        yield

def test_with_fixture(frozen_time):
    """Test using frozen time fixture."""
    assert datetime.now() == datetime(2025, 1, 15, 12, 0, 0)

def test_another_with_fixture(frozen_time):
    """Another test using the same frozen time."""
    assert datetime.now().year == 2025
```

### 4. Class-Level Freezing

```python
@freeze_time("2025-01-15T12:00:00Z")
class TestTimeDependent:
    """All tests in this class use the same frozen time."""

    def test_method_one(self):
        assert datetime.now().year == 2025

    def test_method_two(self):
        assert datetime.now().month == 1
```

## Advanced Features

### Time Progression and Ticking

```python
# Allow time to tick forward at normal speed
@freeze_time("2025-01-15T12:00:00", tick=True)
def test_with_ticking():
    """Time progresses naturally during test."""
    start = datetime.now()
    time.sleep(0.1)  # Actually sleeps
    end = datetime.now()
    assert end > start

# Auto-tick every second
@freeze_time("2025-01-15T12:00:00", auto_tick_seconds=1)
def test_auto_tick():
    """Time automatically advances by 1 second."""
    start = datetime.now()
    # Time automatically progresses
    time.sleep(2)  # Will show 2+ seconds of progress
    end = datetime.now()
    assert (end - start).seconds >= 2
```

### Manual Time Movement

```python
def test_manual_time_movement():
    """Manually control time progression."""
    freezer = freeze_time("2025-01-15T12:00:00")
    freezer.start()

    try:
        # Initial time
        assert datetime.now().hour == 12

        # Move forward 2 hours
        freezer.move_to("2025-01-15T14:00:00")
        assert datetime.now().hour == 14

        # Tick forward by duration
        freezer.tick(delta=timedelta(minutes=30))
        assert datetime.now().hour == 14
        assert datetime.now().minute == 30

    finally:
        freezer.stop()
```

### Timezone Support

```python
from datetime import timezone, timedelta

# UTC timezone
@freeze_time("2025-01-15T12:00:00+00:00")
def test_utc_time():
    now = datetime.now(timezone.utc)
    assert now.hour == 12

# Custom timezone offset
@freeze_time("2025-01-15T12:00:00", tz_offset=-5)
def test_timezone_offset():
    """Test with EST timezone (UTC-5)."""
    now = datetime.now()
    # Time is displayed in local timezone
    assert now.hour == 7  # 12 UTC - 5 hours

# Named timezone
@freeze_time("2025-01-15T12:00:00+00:00")
def test_named_timezone():
    """Test with explicit timezone."""
    utc_time = datetime.now(timezone.utc)
    assert utc_time.tzinfo == timezone.utc
```

## Pytest Integration (pytest-freezegun)

### Using the freezer Fixture

```python
# When pytest-freezegun is installed
def test_with_freezer_fixture(freezer):
    """Use the built-in freezer fixture."""
    freezer.move_to("2025-01-15T12:00:00")
    assert datetime.now() == datetime(2025, 1, 15, 12, 0, 0)

    # Move time forward
    freezer.move_to("2025-01-15T15:30:00")
    assert datetime.now().hour == 15

# Using pytest marks
@pytest.mark.freeze_time('2025-01-15T12:00:00')
def test_with_mark():
    """Test using pytest mark."""
    assert datetime.now() == datetime(2025, 1, 15, 12, 0, 0)
```

## Testing Event Processing with FreezeGun

### Time Slicing Example

```python
@freeze_time("2025-01-15T12:00:00Z")
def test_event_time_slicing():
    """Test event processing with frozen time."""
    events = [
        {"ts": "2025-01-15T12:00:00Z", "kind": "user_action"},
        {"ts": "2025-01-15T12:00:01Z", "kind": "data_update"},
    ]

    # Process events - timestamps are now valid
    result = process_events(events)
    assert len(result.time_slices) == 1  # Both events in same slice
```

### Progressive Time Testing

```python
def test_event_progression():
    """Test event processing over time."""
    with freeze_time("2025-01-15T12:00:00Z") as frozen_time:
        # Process first batch
        events_batch1 = create_events_at_current_time()
        process_events(events_batch1)

        # Move time forward
        frozen_time.move_to("2025-01-15T12:01:00Z")

        # Process second batch
        events_batch2 = create_events_at_current_time()
        process_events(events_batch2)

        # Verify time-based behavior
        assert get_processed_time_slices() == 2
```

## Best Practices (2024-2025)

### 1. Use Relative Times When Possible

```python
from datetime import datetime, timedelta

# Good: Use relative to a fixed point
@freeze_time("2025-01-15T12:00:00Z")
def test_relative_timing():
    base_time = datetime.now()
    future_time = base_time + timedelta(hours=1)
    # Test logic using relative times

# Avoid: Hardcoding current dates that become stale
@freeze_time("2024-12-01")  # Will fail when date passes
def test_bad_practice():
    pass
```

### 2. Localize Time Freezing

```python
def test_mixed_time_behavior():
    """Only freeze time for specific operations."""
    # Normal time behavior
    pre_test_setup()

    # Freeze only for the critical part
    with freeze_time("2025-01-15T12:00:00Z"):
        time_sensitive_operation()

    # Normal time behavior resumes
    post_test_cleanup()
```

### 3. Handle Asyncio Properly (CRITICAL for Async Tests)

```python
# REQUIRED: Use real_asyncio=True for async tests
@freeze_time("2025-01-15T12:00:00Z", real_asyncio=True)
async def test_async_with_freezegun():
    """Use real_asyncio=True for async tests."""
    result = await async_time_dependent_function()
    assert result.timestamp == datetime(2025, 1, 15, 12, 0, 0)

# Without real_asyncio=True, datetime.now() calls in async contexts may not be frozen
@freeze_time("2025-01-15T12:00:00Z")  # BAD: Will fail in async tests
async def test_async_without_real_asyncio():
    """This will likely fail because time isn't properly frozen."""
    # datetime.now() may return real time instead of frozen time
    now = datetime.now(timezone.utc)
    # Assertion may fail unexpectedly
    assert now == datetime(2025, 1, 15, 12, 0, 0)

# Context manager approach for async
async def test_async_context_manager():
    """Async with context manager."""
    with freeze_time("2025-01-15T12:00:00Z", real_asyncio=True):
        result = await async_time_dependent_function()
        assert result.timestamp == datetime(2025, 1, 15, 12, 0, 0)
```

### 4. Clean Test Isolation

```python
class TestTimeDependent:
    """Ensure each test starts with clean time state."""

    def test_first(self):
        with freeze_time("2025-01-15T10:00:00Z"):
            # Test logic
            pass

    def test_second(self):
        # Starts with unfrozen time
        with freeze_time("2025-01-15T14:00:00Z"):
            # Different frozen time
            pass
```

## Common Patterns for CLI Testing

### Event File Testing

```python
@freeze_time("2025-01-15T12:00:00Z")
def test_cli_event_processing():
    """Test CLI with frozen time for valid events."""
    # Create test events with current frozen time
    events = [
        {
            "ts": datetime.now().isoformat() + "Z",
            "kind": "test_event",
            "stream_id": "test_stream",
            "payload": {"test": "data"}
        }
    ]

    # Write to temp file
    with temp_events_file(events) as events_file:
        # Run CLI command - events will be valid
        result = cli_runner.invoke(["process", str(events_file)])
        assert result.exit_code == 0
```

### Configuration Testing

```python
@freeze_time("2025-01-15T12:00:00Z")
def test_config_validation():
    """Test configuration at specific time."""
    config = load_config()

    # Time-based validation will be consistent
    assert config.is_valid_at_time(datetime.now())
```

## Error Prevention

### 1. Avoid Date Boundaries

```python
# Bad: Test might fail at month boundary
@freeze_time("2025-01-31T23:59:59Z")
def test_near_boundary():
    pass

# Good: Use safe dates
@freeze_time("2025-01-15T12:00:00Z")
def test_safe_date():
    pass
```

### 2. Handle Multiple Time Zones

```python
@freeze_time("2025-01-15T00:00:00Z")
def test_timezone_aware():
    """Always use UTC in tests for consistency."""
    utc_time = datetime.now(timezone.utc)
    local_time = datetime.now()

    # Be explicit about timezone expectations
    assert utc_time.tzinfo == timezone.utc
```

## Current Limitations (2025)

1. **Performance**: Slight overhead when time functions are called frequently
2. **Third-party Libraries**: Some libraries may not be affected by freezegun
3. **System Time**: Only affects Python's datetime functions, not system time
4. **Async Compatibility**: Requires `real_asyncio=True` for proper async support

## Integration with Testing Frameworks

Works well with:
- pytest (excellent integration with pytest-freezegun)
- unittest (works with decorators and context managers)
- nose (basic support)
- Custom test runners (via context managers)

This reference should be updated as FreezeGun evolves, particularly regarding asyncio support and performance optimizations.
