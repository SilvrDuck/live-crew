"""Tests to demonstrate the Pydantic generic Event schema fix.

This test suite specifically validates that our pre-generated Event types
work correctly with Freezegun and don't suffer from schema generation issues.
"""

from datetime import datetime, timezone
from freezegun import freeze_time

from tests.utils import EventDict, EventAny, ActionDict, ActionAny


class TestGenericEventFix:
    """Test that our pre-generated generic types work with Freezegun."""

    @freeze_time("2025-07-20T12:00:00Z")
    def test_event_dict_with_freezegun(self):
        """Test EventDict works correctly with Freezegun."""
        event_data = {
            "ts": "2025-07-20T11:55:00Z",
            "kind": "test_event",
            "stream_id": "test_stream",
            "payload": {"test": "data", "number": 42},
        }

        event = EventDict(**event_data)
        assert event.kind == "test_event"
        assert event.stream_id == "test_stream"
        assert event.payload == {"test": "data", "number": 42}
        assert event.ts.tzinfo == timezone.utc

    @freeze_time("2025-07-20T12:00:00Z")
    def test_event_any_with_freezegun(self):
        """Test EventAny works correctly with Freezegun."""
        event_data = {
            "ts": "2025-07-20T11:55:00Z",
            "kind": "flexible_event",
            "stream_id": "test_stream",
            "payload": "can be anything",  # String payload
        }

        event = EventAny(**event_data)
        assert event.kind == "flexible_event"
        assert event.payload == "can be anything"

    @freeze_time("2025-07-20T12:00:00Z")
    def test_action_dict_with_freezegun(self):
        """Test ActionDict works correctly with Freezegun."""
        action_data = {
            "ts": "2025-07-20T11:55:00Z",
            "kind": "test_action",
            "stream_id": "test_stream",
            "payload": {"action": "process", "data": 123},
            "ttl_ms": 10000,
        }

        action = ActionDict(**action_data)
        assert action.kind == "test_action"
        assert action.payload == {"action": "process", "data": 123}
        assert action.ttl_ms == 10000

    @freeze_time("2025-07-20T12:00:00Z")
    def test_action_any_with_freezegun(self):
        """Test ActionAny works correctly with Freezegun."""
        action_data = {
            "ts": "2025-07-20T11:55:00Z",
            "kind": "flexible_action",
            "stream_id": "test_stream",
            "payload": ["list", "of", "items"],  # List payload
        }

        action = ActionAny(**action_data)
        assert action.kind == "flexible_action"
        assert action.payload == ["list", "of", "items"]
        assert action.ttl_ms == 5000  # Default TTL

    def test_schema_generation_timing(self):
        """Test that schema generation happens at import time, not runtime."""
        # This test passes simply by importing successfully without freezegun errors
        # The schemas were already generated when the module was imported

        # We can create instances without any time mocking
        event = EventDict(
            ts=datetime.now(timezone.utc),
            kind="test",
            stream_id="stream",
            payload={"success": True},
        )

        assert event.payload["success"] is True

    @freeze_time("2025-07-20T12:00:00Z")
    def test_multiple_event_types_same_test(self):
        """Test multiple pre-generated types in the same frozen context."""
        base_data = {
            "ts": "2025-07-20T11:55:00Z",
            "kind": "multi_test",
            "stream_id": "test_stream",
        }

        # Different payload types, all should work
        event_dict = EventDict(**{**base_data, "payload": {"type": "dict"}})
        event_any = EventAny(**{**base_data, "payload": "string payload"})

        action_dict = ActionDict(**{**base_data, "payload": {"action": True}})
        action_any = ActionAny(**{**base_data, "payload": 42})

        assert event_dict.payload["type"] == "dict"
        assert event_any.payload == "string payload"
        assert action_dict.payload["action"] is True
        assert action_any.payload == 42
