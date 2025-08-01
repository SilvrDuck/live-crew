"""Tests for Event and Action data models.

Based on specification in .vibes/live_crew_spec.md section 2.1
"""

import pytest
import time
from datetime import datetime, timezone
from typing import Dict, Any
from pydantic import ValidationError
from live_crew.core.models import Event, Action


@pytest.fixture
def fixed_timestamp():
    """Provide a truly fixed timestamp for consistent testing."""
    return datetime(2025, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def event_factory(fixed_timestamp):
    """Factory for creating test events efficiently."""

    def _create_event(
        ts: datetime | None = None,  # type: ignore[misc]  # Factory pattern allows None default
        kind: str = "test_event",
        stream_id: str = "test_stream",
        payload: Any = "test_payload",
    ) -> Event[Any]:
        if ts is None:
            ts = fixed_timestamp
        return Event[Any](ts=ts, kind=kind, stream_id=stream_id, payload=payload)

    return _create_event


@pytest.fixture
def action_factory(fixed_timestamp):
    """Factory for creating test actions efficiently."""

    def _create_action(
        ts: datetime | None = None,  # type: ignore[misc]  # Factory pattern allows None default
        kind: str = "test_action",
        stream_id: str = "test_stream",
        payload: Any = "test_payload",
        ttl_ms: int = 5000,
    ) -> Action[Any]:
        if ts is None:
            ts = fixed_timestamp
        return Action[Any](
            ts=ts, kind=kind, stream_id=stream_id, payload=payload, ttl_ms=ttl_ms
        )

    return _create_action


class TestEvent:
    """Test cases for Event model."""

    def test_event_creation_valid(self):
        """Test creating a valid Event with all required fields."""
        now = datetime.now(timezone.utc)
        payload = {"score": {"home": 1, "away": 0}}

        event = Event[Dict[str, Any]](
            ts=now, kind="goal_scored", stream_id="match42", payload=payload
        )

        assert event.ts == now
        assert event.kind == "goal_scored"
        assert event.stream_id == "match42"
        assert event.payload == payload

    def test_event_immutable(self):
        """Test that Event is immutable (frozen)."""
        event = Event[str](
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="test_stream",
            payload="test_payload",
        )

        with pytest.raises(ValidationError):
            event.kind = "modified"

    def test_event_generic_typing(self):
        """Test Event works with different payload types."""
        # String payload
        str_event = Event[str](
            ts=datetime.now(timezone.utc),
            kind="text_event",
            stream_id="stream1",
            payload="hello world",
        )
        assert isinstance(str_event.payload, str)

        # Dict payload
        dict_event = Event[Dict[str, int]](
            ts=datetime.now(timezone.utc),
            kind="score_event",
            stream_id="stream2",
            payload={"home": 2, "away": 1},
        )
        assert isinstance(dict_event.payload, dict)

    def test_event_validation_missing_fields(self):
        """Test validation fails when required fields are missing."""
        with pytest.raises(ValidationError) as exc_info:
            Event[str](  # type: ignore[misc]  # Intentionally missing required ts parameter
                kind="test_event",
                stream_id="test_stream",
                payload="test",
                # ts missing
            )
        assert "ts" in str(exc_info.value)

    def test_event_validation_invalid_types(self):
        """Test validation fails for invalid field types."""
        with pytest.raises(ValidationError):
            Event[str](
                ts="not_a_datetime",  # type: ignore[arg-type]  # Intentionally wrong type to test validation
                kind="test_event",
                stream_id="test_stream",
                payload="test",
            )

    def test_event_kind_validation(self):
        """Test kind field validation rules."""
        # Valid kind
        event = Event[str](
            ts=datetime.now(timezone.utc),
            kind="valid_event_name",
            stream_id="stream1",
            payload="test",
        )
        assert event.kind == "valid_event_name"

        # Test empty kind
        with pytest.raises(ValidationError):
            Event[str](
                ts=datetime.now(timezone.utc),
                kind="",  # Empty string
                stream_id="stream1",
                payload="test",
            )

    def test_event_stream_id_validation(self):
        """Test stream_id field validation rules."""
        # Valid stream_id
        event = Event[str](
            ts=datetime.now(timezone.utc),
            kind="test_event",
            stream_id="valid_stream_123",
            payload="test",
        )
        assert event.stream_id == "valid_stream_123"

        # Test empty stream_id
        with pytest.raises(ValidationError):
            Event[str](
                ts=datetime.now(timezone.utc),
                kind="test_event",
                stream_id="",  # Empty string
                payload="test",
            )

    def test_event_json_serialization(self, fixed_timestamp):
        """Test Event can be serialized to/from JSON."""
        original = Event[Dict[str, Any]](
            ts=fixed_timestamp,
            kind="goal_scored",
            stream_id="match42",
            payload={"team": "home", "player": "Messi"},
        )

        # Serialize to JSON
        json_str = original.model_dump_json()
        assert "goal_scored" in json_str
        assert "match42" in json_str

        # Deserialize from JSON
        parsed = Event[Dict[str, Any]].model_validate_json(json_str)
        assert parsed.kind == original.kind
        assert parsed.stream_id == original.stream_id
        assert parsed.payload == original.payload


class TestAction:
    """Test cases for Action model."""

    def test_action_creation_valid(self):
        """Test creating a valid Action with all fields."""
        now = datetime.now(timezone.utc)
        payload = {"text": "GOAL! Amazing strike!"}

        action = Action[Dict[str, str]](
            ts=now,
            kind="commentary_line",
            stream_id="match42",
            payload=payload,
            ttl_ms=10000,
        )

        assert action.ts == now
        assert action.kind == "commentary_line"
        assert action.stream_id == "match42"
        assert action.payload == payload
        assert action.ttl_ms == 10000

    def test_action_default_ttl(self):
        """Test Action uses default ttl_ms of 5000."""
        action = Action[str](
            ts=datetime.now(timezone.utc),
            kind="test_action",
            stream_id="test_stream",
            payload="test",
            # ttl_ms not specified - should default to 5000
        )

        assert action.ttl_ms == 5000

    def test_action_immutable(self):
        """Test that Action is immutable (frozen)."""
        action = Action[str](
            ts=datetime.now(timezone.utc),
            kind="test_action",
            stream_id="test_stream",
            payload="test_payload",
        )

        with pytest.raises(ValidationError):
            action.ttl_ms = 10000

    def test_action_ttl_validation(self):
        """Test ttl_ms field validation."""
        # Valid positive ttl_ms
        action = Action[str](
            ts=datetime.now(timezone.utc),
            kind="test_action",
            stream_id="stream1",
            payload="test",
            ttl_ms=1000,
        )
        assert action.ttl_ms == 1000

        # Invalid negative ttl_ms
        with pytest.raises(ValidationError):
            Action[str](
                ts=datetime.now(timezone.utc),
                kind="test_action",
                stream_id="stream1",
                payload="test",
                ttl_ms=-1000,  # Negative
            )

        # Invalid zero ttl_ms
        with pytest.raises(ValidationError):
            Action[str](
                ts=datetime.now(timezone.utc),
                kind="test_action",
                stream_id="stream1",
                payload="test",
                ttl_ms=0,  # Zero
            )

    def test_action_generic_typing(self):
        """Test Action works with different payload types."""
        # Complex payload
        complex_payload = {
            "commentary": "What a goal!",
            "language": "en-US",
            "confidence": 0.95,
        }

        action = Action[Dict[str, Any]](
            ts=datetime.now(timezone.utc),
            kind="ai_commentary",
            stream_id="match42",
            payload=complex_payload,
        )

        assert action.payload == complex_payload
        assert isinstance(action.payload, dict)

    def test_action_validation_missing_fields(self):
        """Test validation fails when required fields are missing."""
        with pytest.raises(ValidationError) as exc_info:
            Action[str](
                ts=datetime.now(timezone.utc),
                kind="test_action",
                # stream_id missing
                payload="test",
            )
        assert "stream_id" in str(exc_info.value)

    def test_action_json_serialization(self, fixed_timestamp):
        """Test Action can be serialized to/from JSON."""
        original = Action[Dict[str, Any]](
            ts=fixed_timestamp,
            kind="score_update",
            stream_id="match42",
            payload={"home": 2, "away": 1},
            ttl_ms=8000,
        )

        # Serialize to JSON
        json_str = original.model_dump_json()
        assert "score_update" in json_str
        assert "8000" in json_str

        # Deserialize from JSON
        parsed = Action[Dict[str, Any]].model_validate_json(json_str)
        assert parsed.kind == original.kind
        assert parsed.ttl_ms == original.ttl_ms
        assert parsed.payload == original.payload


class TestEventActionComparison:
    """Test interactions between Event and Action models."""

    def test_event_action_same_structure(self):
        """Test that Event and Action share common fields."""
        now = datetime.now(timezone.utc)

        event = Event[str](
            ts=now, kind="test_message", stream_id="stream1", payload="test_data"
        )

        action = Action[str](
            ts=now, kind="test_message", stream_id="stream1", payload="test_data"
        )

        # Common fields should match
        assert event.ts == action.ts
        assert event.kind == action.kind
        assert event.stream_id == action.stream_id
        assert event.payload == action.payload

        # Action has additional ttl_ms field
        assert hasattr(action, "ttl_ms")
        assert not hasattr(event, "ttl_ms")

    def test_event_action_different_payloads(self):
        """Test Event and Action can have different payload types."""
        event = Event[str](
            ts=datetime.now(timezone.utc),
            kind="text_input",
            stream_id="stream1",
            payload="input text",
        )

        action = Action[Dict[str, int]](
            ts=datetime.now(timezone.utc),
            kind="score_output",
            stream_id="stream1",
            payload={"home": 1, "away": 0},
        )

        assert isinstance(event.payload, str)
        assert isinstance(action.payload, dict)


class TestEventEdgeCases:
    """Test Event edge cases and boundary conditions."""

    def test_event_max_length_validation(self, fixed_timestamp):
        """Test maximum length constraints for kind and stream_id."""
        # Test kind max length (should be 50 characters)
        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="x" * 51,  # Exceeds max_length=50
                stream_id="stream1",
                payload="test",
            )

        # Test stream_id max length (should be 100 characters)
        with pytest.raises(ValidationError):
            Event[str](
                ts=fixed_timestamp,
                kind="test",
                stream_id="x" * 101,  # Exceeds max_length=100
                payload="test",
            )

    def test_event_timezone_handling(self, fixed_timestamp):
        """Test timezone handling and UTC conversion."""
        # Test naive datetime (should be converted to UTC) - use fixed time as naive
        naive_dt = fixed_timestamp.replace(
            tzinfo=None
        )  # Remove timezone info but keep the time

        event = Event[str](
            ts=naive_dt, kind="test_event", stream_id="stream1", payload="test"
        )
        assert event.ts.tzinfo == timezone.utc

        # Test future timestamp validation (should fail) - use a very far future time
        very_future_dt = datetime(2030, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValidationError):
            Event[str](
                ts=very_future_dt,
                kind="test_event",
                stream_id="stream1",
                payload="test",
            )

        # Test very old timestamp validation (should fail) - use a very old time
        very_old_dt = datetime(1990, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValidationError):
            Event[str](
                ts=very_old_dt, kind="test_event", stream_id="stream1", payload="test"
            )


class TestActionEdgeCases:
    """Test Action edge cases and boundary conditions."""

    def test_action_ttl_boundary_values(self, fixed_timestamp):
        """Test TTL boundary validation."""
        # Test minimum valid TTL (1ms)
        action = Action[str](
            ts=fixed_timestamp,
            kind="test",
            stream_id="stream1",
            payload="test",
            ttl_ms=1,
        )
        assert action.ttl_ms == 1

        # Test maximum valid TTL (300,000ms = 5 minutes)
        action = Action[str](
            ts=fixed_timestamp,
            kind="test",
            stream_id="stream1",
            payload="test",
            ttl_ms=300_000,
        )
        assert action.ttl_ms == 300_000

        # Test exceeding maximum TTL
        with pytest.raises(ValidationError):
            Action[str](
                ts=fixed_timestamp,
                kind="test",
                stream_id="stream1",
                payload="test",
                ttl_ms=300_001,
            )

        # Test zero TTL (should fail)
        with pytest.raises(ValidationError):
            Action[str](
                ts=fixed_timestamp,
                kind="test",
                stream_id="stream1",
                payload="test",
                ttl_ms=0,
            )


class TestPerformance:
    """Test performance characteristics for high-frequency processing."""

    def test_event_creation_performance(self, fixed_timestamp):
        """Test Event creation performance for high-frequency scenarios."""
        start_time = time.perf_counter()

        # Create 1000 events (simulating high frequency)
        events = []
        for i in range(1000):
            event = Event[Dict[str, int]](
                ts=fixed_timestamp,
                kind="perf_test",
                stream_id=f"stream_{i}",
                payload={"value": i},
            )
            events.append(event)

        duration = time.perf_counter() - start_time

        # Should create 1000 events in less than 100ms for real-time processing
        assert duration < 0.1, (
            f"Event creation too slow: {duration:.3f}s for 1000 events"
        )
        assert len(events) == 1000

    def test_json_serialization_performance(self, event_factory):
        """Test JSON serialization performance."""
        # Create test event with complex payload
        event = event_factory(
            payload={
                "nested": {"data": [1, 2, 3, 4, 5]},
                "text": "Sample event data for serialization testing",
                "numbers": list(range(100)),
            }
        )

        start_time = time.perf_counter()

        # Serialize 1000 times
        for _ in range(1000):
            _ = event.model_dump_json(by_alias=True, exclude_unset=True)

        duration = time.perf_counter() - start_time

        # Should serialize 1000 times in less than 50ms
        assert duration < 0.05, (
            f"JSON serialization too slow: {duration:.3f}s for 1000 operations"
        )

    def test_model_validation_performance(self, fixed_timestamp):
        """Test model validation performance under load."""
        start_time = time.perf_counter()

        # Test data
        test_data = {
            "ts": fixed_timestamp,
            "kind": "performance_test",
            "stream_id": "perf_stream",
            "payload": {"test": "data"},
        }

        # Validate 1000 times
        events = []
        for i in range(1000):
            test_data["payload"]["index"] = i  # type: ignore[index]  # We know payload is a dict
            event = Event[Dict[str, Any]](
                ts=test_data["ts"],  # type: ignore[arg-type]  # We know this is a datetime
                kind=test_data["kind"],  # type: ignore[arg-type]  # We know this is a string
                stream_id=test_data["stream_id"],  # type: ignore[arg-type]  # We know this is a string
                payload=test_data["payload"],  # type: ignore[arg-type]  # We know this is a dict
            )
            events.append(event)

        duration = time.perf_counter() - start_time

        # Should validate 1000 events in less than 50ms
        assert duration < 0.05, (
            f"Model validation too slow: {duration:.3f}s for 1000 validations"
        )
        assert len(events) == 1000
