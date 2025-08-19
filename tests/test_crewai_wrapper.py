"""Tests for CrewAI wrapper that adapts CrewAI crews to EventHandler protocol.

Comprehensive test coverage for CrewAIWrapper including event handling,
data conversion, error scenarios, and integration with mock CrewAI crews.
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import Mock

from live_crew.core.models import Event
from live_crew.crewai_integration.wrapper import (
    CrewAIWrapper,
    CrewAIExecutionError,
    CrewAIContextBridge,
)


@pytest.fixture
def fixed_timestamp():
    """Provide a fixed timestamp for consistent testing."""
    return datetime(2025, 8, 6, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_crewai_crew():
    """Create a mock CrewAI crew for testing."""
    crew = Mock()
    crew.kickoff = Mock()
    return crew


@pytest.fixture
def sample_event(fixed_timestamp):
    """Create a sample event for testing."""
    return Event[Dict[str, Any]](
        ts=fixed_timestamp,
        kind="user_input",
        stream_id="test_stream_123",
        payload={
            "message": "Hello world",
            "user_id": "user_42",
            "metadata": {"priority": "high", "source": "web_app"},
        },
    )


@pytest.fixture
def sample_context():
    """Create sample context data for testing."""
    return {
        "session_id": "sess_789",
        "user_preferences": {"theme": "dark", "language": "en"},
        "current_slice": 42,
        "global_state": {"system_status": "active"},
    }


class TestCrewAIWrapperInitialization:
    """Test cases for CrewAIWrapper initialization."""

    def test_wrapper_init_valid_minimal(self, mock_crewai_crew):
        """Test wrapper initialization with minimal valid parameters."""
        wrapper = CrewAIWrapper(
            crew_id="test_crew", crewai_crew=mock_crewai_crew, triggers=["user_input"]
        )

        assert wrapper.crew_id == "test_crew"
        assert wrapper.crewai_crew == mock_crewai_crew
        assert wrapper.triggers == ["user_input"]
        assert wrapper.timeout_ms == 5000  # Default timeout

    def test_wrapper_init_valid_full(self, mock_crewai_crew):
        """Test wrapper initialization with all parameters specified."""
        triggers = ["event1", "event2", "event3"]
        timeout = 15000

        wrapper = CrewAIWrapper(
            crew_id="analysis_crew",
            crewai_crew=mock_crewai_crew,
            triggers=triggers,
            timeout_ms=timeout,
        )

        assert wrapper.crew_id == "analysis_crew"
        assert wrapper.crewai_crew == mock_crewai_crew
        assert wrapper.triggers == triggers
        assert wrapper.timeout_ms == timeout

    def test_wrapper_init_empty_triggers(self, mock_crewai_crew):
        """Test wrapper initialization with empty triggers list."""
        wrapper = CrewAIWrapper(
            crew_id="test_crew", crewai_crew=mock_crewai_crew, triggers=[]
        )

        assert wrapper.triggers == []

    def test_wrapper_init_none_crew(self):
        """Test wrapper initialization with None crew."""
        wrapper = CrewAIWrapper(
            crew_id="test_crew",
            crewai_crew=None,  # type: ignore  # Testing edge case
            triggers=["test"],
        )

        assert wrapper.crewai_crew is None

    def test_wrapper_init_boundary_timeout_values(self, mock_crewai_crew):
        """Test wrapper initialization with boundary timeout values."""
        # Very small timeout
        wrapper_small = CrewAIWrapper(
            crew_id="test_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["test"],
            timeout_ms=1,
        )
        assert wrapper_small.timeout_ms == 1

        # Very large timeout
        wrapper_large = CrewAIWrapper(
            crew_id="test_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["test"],
            timeout_ms=300000,
        )
        assert wrapper_large.timeout_ms == 300000

    def test_wrapper_init_special_crew_id_formats(self, mock_crewai_crew):
        """Test wrapper initialization with various crew ID formats."""
        special_ids = [
            "simple_crew",
            "crew-with-dashes",
            "crew.with.dots",
            "UPPERCASE_CREW",
            "123_numeric_start",
            "crew_123_numbers",
            "very_long_crew_id_with_many_underscores_and_numbers_123_456",
            "",  # Empty string
        ]

        for crew_id in special_ids:
            wrapper = CrewAIWrapper(
                crew_id=crew_id, crewai_crew=mock_crewai_crew, triggers=["test"]
            )
            assert wrapper.crew_id == crew_id


class TestCrewAIWrapperEventHandling:
    """Test cases for CrewAI wrapper event handling."""

    @pytest.mark.asyncio
    async def test_handle_event_string_result(
        self, sample_event, sample_context, mock_crewai_crew, fixed_timestamp
    ):
        """Test handling event with CrewAI returning string result."""
        # Setup mock to return string
        mock_crewai_crew.kickoff.return_value = (
            "Analysis complete: The user message indicates a greeting."
        )

        wrapper = CrewAIWrapper(
            crew_id="text_analyzer",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
            timeout_ms=10000,
        )

        # Handle the event
        actions = await wrapper.handle_event(sample_event, sample_context)

        # Verify crew was called
        mock_crewai_crew.kickoff.assert_called_once()

        # Verify actions
        assert len(actions) == 1
        action = actions[0]

        assert action.kind == "text_analyzer_text"
        assert action.stream_id == "test_stream_123"
        assert (
            action.payload["text"]
            == "Analysis complete: The user message indicates a greeting."
        )
        assert action.payload["source_event"] == "user_input"
        assert action.ttl_ms == 10000

    @pytest.mark.asyncio
    async def test_handle_event_dict_result(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test handling event with CrewAI returning dict result."""
        # Setup mock to return dictionary
        mock_result = {
            "analysis": "greeting detected",
            "sentiment": "positive",
            "confidence": 0.95,
            "entities": ["user_42"],
        }
        mock_crewai_crew.kickoff.return_value = mock_result

        wrapper = CrewAIWrapper(
            crew_id="sentiment_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
        )

        actions = await wrapper.handle_event(sample_event, sample_context)

        assert len(actions) == 1
        action = actions[0]

        assert action.kind == "sentiment_crew_structured"
        assert action.payload["data"] == mock_result
        assert action.payload["source_event"] == "user_input"

    @pytest.mark.asyncio
    async def test_handle_event_crew_result_object(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test handling event with CrewAI returning CrewResult object."""
        # Create mock CrewResult with raw attribute
        mock_crew_result = Mock()
        mock_crew_result.raw = "Processed successfully with detailed analysis"
        mock_crewai_crew.kickoff.return_value = mock_crew_result

        wrapper = CrewAIWrapper(
            crew_id="processing_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
        )

        actions = await wrapper.handle_event(sample_event, sample_context)

        assert len(actions) == 1
        action = actions[0]

        assert action.kind == "processing_crew_output"
        assert (
            action.payload["crew_result"]
            == "Processed successfully with detailed analysis"
        )
        assert action.payload["source_event"] == "user_input"

    @pytest.mark.asyncio
    async def test_handle_event_generic_result(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test handling event with CrewAI returning non-standard result type."""
        # Setup mock to return a custom object
        mock_result = ["item1", "item2", 42, {"nested": "data"}]
        mock_crewai_crew.kickoff.return_value = mock_result

        wrapper = CrewAIWrapper(
            crew_id="list_processor",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
        )

        actions = await wrapper.handle_event(sample_event, sample_context)

        assert len(actions) == 1
        action = actions[0]

        assert action.kind == "list_processor_generic"
        # Result should be JSON serialized
        expected_result = json.dumps(mock_result, default=str)
        assert action.payload["result"] == expected_result

    @pytest.mark.asyncio
    async def test_handle_event_unserializable_result(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test handling event with CrewAI returning unserializable result."""

        # Create an object that can't be JSON serialized
        class UnserializableObject:
            def __str__(self):
                return "UnserializableObject instance"

        mock_result = UnserializableObject()
        # Make JSON serialization fail
        mock_crewai_crew.kickoff.return_value = mock_result

        wrapper = CrewAIWrapper(
            crew_id="fallback_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
        )

        actions = await wrapper.handle_event(sample_event, sample_context)

        assert len(actions) == 1
        action = actions[0]

        assert action.kind == "fallback_crew_generic"
        # The object gets serialized as JSON with default=str, so it's a JSON string
        import json

        expected_result = json.dumps(mock_result, default=str)
        assert action.payload["result"] == expected_result

    @pytest.mark.asyncio
    async def test_handle_event_crew_execution_error(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test handling event when CrewAI crew execution fails."""
        # Setup mock to raise exception
        mock_crewai_crew.kickoff.side_effect = Exception("CrewAI internal error")

        wrapper = CrewAIWrapper(
            crew_id="failing_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["user_input"],
        )

        with pytest.raises(CrewAIExecutionError) as exc_info:
            await wrapper.handle_event(sample_event, sample_context)

        error_msg = str(exc_info.value)
        assert "failing_crew" in error_msg
        assert "failed during event processing" in error_msg
        assert "CrewAI internal error" in error_msg

    @pytest.mark.asyncio
    async def test_handle_event_none_crew(self, sample_event, sample_context):
        """Test handling event with None crew raises appropriate error."""
        wrapper = CrewAIWrapper(
            crew_id="none_crew",
            crewai_crew=None,  # type: ignore  # Testing edge case
            triggers=["user_input"],
        )

        with pytest.raises(CrewAIExecutionError):
            await wrapper.handle_event(sample_event, sample_context)


class TestCrewAIWrapperInputPreparation:
    """Test cases for CrewAI wrapper input preparation."""

    def test_prepare_crew_inputs_simple_payload(self, fixed_timestamp):
        """Test preparing crew inputs with simple payload."""
        event = Event[str](
            ts=fixed_timestamp,
            kind="text_input",
            stream_id="stream_1",
            payload="Simple text message",
        )

        context = {"session": "123", "user": "alice"}
        context_bridge = CrewAIContextBridge(context, "test_crew")

        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])
        inputs = wrapper._prepare_crew_inputs(event, context_bridge)

        # Verify basic event metadata
        assert inputs["event_kind"] == "text_input"
        assert inputs["event_timestamp"] == fixed_timestamp.isoformat()
        assert inputs["stream_id"] == "stream_1"

        # Verify payload handling
        assert inputs["payload"] == "Simple text message"

        # Verify context handling through context bridge
        assert inputs["session"] == "123"
        assert inputs["user"] == "alice"

        # Verify live_crew nested context is available
        assert "live_crew" in inputs
        assert inputs["live_crew"]["crew_id"] == "test_crew"
        assert inputs["live_crew"]["context"]["session"] == "123"

    def test_prepare_crew_inputs_dict_payload(self, fixed_timestamp):
        """Test preparing crew inputs with dictionary payload."""
        payload = {
            "message": "Hello",
            "priority": "high",
            "nested": {"level": "deep", "data": "important"},
        }

        event = Event[Dict[str, Any]](
            ts=fixed_timestamp,
            kind="structured_input",
            stream_id="stream_2",
            payload=payload,
        )

        context = {"environment": "production"}
        context_bridge = CrewAIContextBridge(context, "test_crew")

        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])
        inputs = wrapper._prepare_crew_inputs(event, context_bridge)

        # Verify flattened payload
        assert inputs["message"] == "Hello"
        assert inputs["priority"] == "high"
        assert inputs["nested_level"] == "deep"
        assert inputs["nested_data"] == "important"

        # Verify context
        assert inputs["environment"] == "production"

    def test_prepare_crew_inputs_complex_nested_payload(self, fixed_timestamp):
        """Test preparing crew inputs with complex nested structures."""
        payload = {
            "user_data": {
                "name": "John Doe",
                "preferences": {"theme": "dark", "notifications": True},
            },
            "request_info": {
                "endpoint": "/api/v1/process",
                "method": "POST",
                "timestamp": "2025-08-06T12:00:00Z",
            },
            "simple_field": "value",
        }

        event = Event[Dict[str, Any]](
            ts=fixed_timestamp,
            kind="api_request",
            stream_id="api_stream",
            payload=payload,
        )

        context = {"api_version": "v1", "rate_limit": {"requests": 100, "window": 60}}
        context_bridge = CrewAIContextBridge(context, "api_crew")

        wrapper = CrewAIWrapper("api_crew", Mock(), ["api_request"])
        inputs = wrapper._prepare_crew_inputs(event, context_bridge)

        # Verify nested flattening (only one level deep)
        assert inputs["user_data_name"] == "John Doe"
        # Note: nested.preferences stays as dict since we only flatten one level
        assert inputs["user_data_preferences"] == {
            "theme": "dark",
            "notifications": True,
        }
        assert inputs["request_info_endpoint"] == "/api/v1/process"
        assert inputs["request_info_method"] == "POST"
        assert inputs["simple_field"] == "value"

        # Verify context with nested structures
        assert inputs["api_version"] == "v1"
        assert inputs["rate_limit"] == {"requests": 100, "window": 60}

    def test_prepare_crew_inputs_empty_context(self, fixed_timestamp):
        """Test preparing crew inputs with empty context."""
        event = Event[str](
            ts=fixed_timestamp,
            kind="test_event",
            stream_id="test_stream",
            payload="test payload",
        )

        context = {}
        context_bridge = CrewAIContextBridge(context, "test_crew")

        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])
        inputs = wrapper._prepare_crew_inputs(event, context_bridge)

        # Verify empty context still provides live_crew metadata
        assert "live_crew" in inputs
        assert inputs["live_crew"]["crew_id"] == "test_crew"
        # Should not have any ctx_ prefixed keys with new approach
        ctx_keys = [k for k in inputs.keys() if k.startswith("ctx_")]
        assert len(ctx_keys) == 0

    def test_prepare_crew_inputs_special_context_keys(self, fixed_timestamp):
        """Test preparing crew inputs with special context key names."""
        event = Event[str](
            ts=fixed_timestamp,
            kind="test_event",
            stream_id="test_stream",
            payload="test",
        )

        context = {
            "event_kind": "conflicting_key",  # Conflicts with event metadata
            "stream_id": "conflicting_stream",  # Conflicts with event metadata
            "payload": "conflicting_payload",  # Conflicts with payload
            "special-chars": "value",
            "numbers_123": "numeric",
            "": "empty_key",  # Edge case
            "very_long_key_name_with_many_words": "long_value",
        }

        context_bridge = CrewAIContextBridge(context, "test_crew")

        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])
        inputs = wrapper._prepare_crew_inputs(event, context_bridge)

        # Event metadata should take precedence over context
        assert inputs["event_kind"] == "test_event"  # From event, not context
        assert inputs["stream_id"] == "test_stream"  # From event, not context
        assert inputs["payload"] == "test"  # From event, not context

        # Non-conflicting context values are available directly
        assert inputs["special-chars"] == "value"
        assert inputs["numbers_123"] == "numeric"
        assert inputs[""] == "empty_key"
        assert inputs["very_long_key_name_with_many_words"] == "long_value"

        # Conflicting context values are available in live_crew nested structure
        assert inputs["live_crew"]["context"]["event_kind"] == "conflicting_key"
        assert inputs["live_crew"]["context"]["stream_id"] == "conflicting_stream"
        assert inputs["live_crew"]["context"]["payload"] == "conflicting_payload"

    def test_flatten_payload_non_dict(self):
        """Test flattening non-dictionary payloads."""
        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])

        # Test string payload
        result = wrapper._flatten_payload("simple string")
        assert result == {"payload": "simple string"}

        # Test numeric payload
        result = wrapper._flatten_payload(42)
        assert result == {"payload": 42}

        # Test boolean payload
        result = wrapper._flatten_payload(True)
        assert result == {"payload": True}

        # Test None payload
        result = wrapper._flatten_payload(None)
        assert result == {"payload": None}

        # Test list payload
        result = wrapper._flatten_payload([1, 2, 3])
        assert result == {"payload": [1, 2, 3]}

    def test_flatten_payload_empty_dict(self):
        """Test flattening empty dictionary payload."""
        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])

        result = wrapper._flatten_payload({})
        assert result == {}

    def test_flatten_payload_mixed_value_types(self):
        """Test flattening payload with mixed value types."""
        payload = {
            "string_val": "text",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None,
            "list_val": [1, 2, 3],
            "nested_dict": {"inner_string": "inner_text", "inner_number": 100},
        }

        wrapper = CrewAIWrapper("test_crew", Mock(), ["test"])
        result = wrapper._flatten_payload(payload)

        assert result["string_val"] == "text"
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["null_val"] is None
        assert result["list_val"] == [1, 2, 3]
        assert result["nested_dict_inner_string"] == "inner_text"
        assert result["nested_dict_inner_number"] == 100


class TestCrewAIWrapperActionConversion:
    """Test cases for CrewAI wrapper action conversion."""

    def test_convert_to_actions_preserves_stream_and_metadata(
        self, sample_event, mock_crewai_crew
    ):
        """Test that action conversion preserves stream ID and event metadata."""
        wrapper = CrewAIWrapper(
            crew_id="metadata_crew",
            crewai_crew=mock_crewai_crew,
            triggers=["test"],
            timeout_ms=15000,
        )

        # Test with different result types
        results = ["string result", {"key": "value"}, Mock(raw="crew result object")]

        for result in results:
            actions = wrapper._convert_to_actions(result, sample_event)

            assert len(actions) == 1
            action = actions[0]

            # Verify metadata preservation
            assert action.stream_id == sample_event.stream_id
            assert action.payload["source_event"] == sample_event.kind
            assert action.ttl_ms == 15000
            # Timestamp should be close to now
            assert (datetime.now(timezone.utc) - action.ts).total_seconds() < 1

    def test_convert_to_actions_kind_generation(self, sample_event, mock_crewai_crew):
        """Test that action kinds are generated correctly based on result type."""
        wrapper = CrewAIWrapper("kind_test_crew", mock_crewai_crew, ["test"])

        # String result
        actions = wrapper._convert_to_actions("text result", sample_event)
        assert actions[0].kind == "kind_test_crew_text"

        # Dict result
        actions = wrapper._convert_to_actions({"data": "value"}, sample_event)
        assert actions[0].kind == "kind_test_crew_structured"

        # CrewResult object
        crew_result = Mock(raw="crew output")
        actions = wrapper._convert_to_actions(crew_result, sample_event)
        assert actions[0].kind == "kind_test_crew_output"

        # Generic result
        actions = wrapper._convert_to_actions([1, 2, 3], sample_event)
        assert actions[0].kind == "kind_test_crew_generic"

        # Generic result (serializable with default=str)
        class SerializableClass:
            def __str__(self):
                return "fallback_string"

        actions = wrapper._convert_to_actions(SerializableClass(), sample_event)
        assert actions[0].kind == "kind_test_crew_generic"

    def test_convert_to_actions_payload_structure(self, sample_event, mock_crewai_crew):
        """Test that action payloads are structured correctly."""
        wrapper = CrewAIWrapper("payload_crew", mock_crewai_crew, ["test"])

        # Test string result payload
        actions = wrapper._convert_to_actions("string result", sample_event)
        payload = actions[0].payload
        assert payload["text"] == "string result"
        assert payload["source_event"] == sample_event.kind

        # Test dict result payload
        dict_result = {"analysis": "complete", "score": 0.95}
        actions = wrapper._convert_to_actions(dict_result, sample_event)
        payload = actions[0].payload
        assert payload["data"] == dict_result
        assert payload["source_event"] == sample_event.kind

        # Test CrewResult payload
        crew_result = Mock(raw={"detailed": "output"})
        actions = wrapper._convert_to_actions(crew_result, sample_event)
        payload = actions[0].payload
        assert payload["crew_result"] == {"detailed": "output"}
        assert payload["source_event"] == sample_event.kind


class TestCrewAIWrapperEdgeCases:
    """Test edge cases and boundary conditions for CrewAI wrapper."""

    @pytest.mark.asyncio
    async def test_handle_event_with_different_event_types(
        self, mock_crewai_crew, sample_context, fixed_timestamp
    ):
        """Test handling events with different payload types."""
        mock_crewai_crew.kickoff.return_value = "processed"

        wrapper = CrewAIWrapper("versatile_crew", mock_crewai_crew, ["any_event"])

        # Test different event payload types
        event_types = [
            Event[str](ts=fixed_timestamp, kind="text", stream_id="s1", payload="text"),
            Event[int](ts=fixed_timestamp, kind="number", stream_id="s2", payload=42),
            Event[bool](
                ts=fixed_timestamp, kind="boolean", stream_id="s3", payload=True
            ),
            Event[List[int]](
                ts=fixed_timestamp, kind="list", stream_id="s4", payload=[1, 2, 3]
            ),
            Event[None](ts=fixed_timestamp, kind="null", stream_id="s5", payload=None),
        ]

        for event in event_types:
            actions = await wrapper.handle_event(event, sample_context)
            assert len(actions) == 1
            assert actions[0].kind == "versatile_crew_text"
            assert actions[0].stream_id == event.stream_id

    @pytest.mark.asyncio
    async def test_handle_event_with_extreme_context_values(
        self, sample_event, mock_crewai_crew
    ):
        """Test handling events with extreme context values."""
        mock_crewai_crew.kickoff.return_value = "handled extreme context"

        # Create context with extreme values
        extreme_context = {
            "empty_string": "",
            "very_long_string": "x" * 10000,
            "nested_deep": {"level1": {"level2": {"level3": {"data": "deep"}}}},
            "large_number": 9999999999999999,
            "negative_number": -9999999999999999,
            "float_precision": 3.141592653589793238462643383279,
            "unicode_chars": "测试数据 🚀 émojis and spëciäl chârs",
            "null_value": None,
            "empty_list": [],
            "empty_dict": {},
            "mixed_list": ["string", 42, True, None, {"nested": "in_list"}],
        }

        wrapper = CrewAIWrapper("extreme_crew", mock_crewai_crew, ["user_input"])

        # Should handle extreme context without errors
        actions = await wrapper.handle_event(sample_event, extreme_context)
        assert len(actions) == 1

        # Verify crew was called with inputs
        mock_crewai_crew.kickoff.assert_called_once()
        call_args = mock_crewai_crew.kickoff.call_args[1]["inputs"]

        # Verify extreme context values were included directly
        assert call_args["empty_string"] == ""
        assert call_args["very_long_string"] == "x" * 10000
        assert call_args["unicode_chars"] == "测试数据 🚀 émojis and spëciäl chârs"

    @pytest.mark.asyncio
    async def test_handle_event_concurrent_execution_simulation(
        self, mock_crewai_crew, fixed_timestamp
    ):
        """Test wrapper behavior under simulated concurrent execution."""
        # Simulate different execution times and results
        import asyncio

        def slow_kickoff(*args, **kwargs):
            # Simulate different results based on inputs
            inputs = kwargs.get("inputs", {})
            stream_id = inputs.get("stream_id", "unknown")
            return f"Result for {stream_id}"

        mock_crewai_crew.kickoff = slow_kickoff

        wrapper = CrewAIWrapper("concurrent_crew", mock_crewai_crew, ["test_event"])

        # Create multiple events
        events = [
            Event[str](
                ts=fixed_timestamp,
                kind="test_event",
                stream_id=f"stream_{i}",
                payload=f"payload_{i}",
            )
            for i in range(5)
        ]

        context = {"test": "concurrent"}

        # Process events concurrently
        tasks = [wrapper.handle_event(event, context) for event in events]
        results = await asyncio.gather(*tasks)

        # Verify all events were processed
        assert len(results) == 5
        for i, actions in enumerate(results):
            assert len(actions) == 1
            action = actions[0]
            assert action.stream_id == f"stream_{i}"
            assert f"stream_{i}" in action.payload["text"]

    def test_wrapper_attributes_immutability(self, mock_crewai_crew):
        """Test that wrapper attributes can be modified (not immutable by design)."""
        wrapper = CrewAIWrapper("mutable_crew", mock_crewai_crew, ["test"])

        # Wrapper attributes should be mutable for flexibility
        original_timeout = wrapper.timeout_ms
        wrapper.timeout_ms = 20000
        assert wrapper.timeout_ms == 20000
        assert wrapper.timeout_ms != original_timeout

        # Triggers should be mutable
        wrapper.triggers.append("new_trigger")
        assert "new_trigger" in wrapper.triggers

    def test_wrapper_with_malformed_crew_result(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test wrapper handling of malformed CrewAI results."""

        # Create a mock result that has 'raw' attribute but it's not accessible
        class MalformedResult:
            @property
            def raw(self):
                raise AttributeError("Cannot access raw attribute")

        malformed_result = MalformedResult()
        mock_crewai_crew.kickoff.return_value = malformed_result

        wrapper = CrewAIWrapper("malformed_crew", mock_crewai_crew, ["user_input"])

        # Should handle the malformed result gracefully by falling back to generic handling
        actions = wrapper._convert_to_actions(malformed_result, sample_event)

        assert len(actions) == 1
        action = actions[0]
        # Should fall back to generic handling since 'raw' access failed but object is still serializable
        assert action.kind == "malformed_crew_generic"


class TestCrewAIExecutionError:
    """Test cases for CrewAIExecutionError exception."""

    def test_crew_ai_execution_error_basic(self):
        """Test basic CrewAIExecutionError creation and attributes."""
        error_msg = "Test error message"
        error = CrewAIExecutionError(error_msg)

        assert str(error) == error_msg
        assert isinstance(error, Exception)

    def test_crew_ai_execution_error_with_cause(self):
        """Test CrewAIExecutionError with cause chain."""
        original_error = ValueError("Original problem")

        try:
            raise original_error
        except ValueError as e:
            crew_error = CrewAIExecutionError("Crew execution failed")
            crew_error.__cause__ = e

        assert str(crew_error) == "Crew execution failed"
        assert crew_error.__cause__ == original_error

    @pytest.mark.asyncio
    async def test_crew_ai_execution_error_in_context(
        self, sample_event, sample_context, mock_crewai_crew
    ):
        """Test CrewAIExecutionError raised in actual wrapper context."""
        # Setup crew to raise various types of exceptions
        exception_types = [
            ValueError("Invalid input"),
            RuntimeError("Runtime issue"),
            KeyError("Missing key"),
            AttributeError("Missing attribute"),
            TypeError("Type mismatch"),
            Exception("Generic exception"),
        ]

        wrapper = CrewAIWrapper("error_crew", mock_crewai_crew, ["user_input"])

        for original_exception in exception_types:
            mock_crewai_crew.kickoff.side_effect = original_exception

            with pytest.raises(CrewAIExecutionError) as exc_info:
                await wrapper.handle_event(sample_event, sample_context)

            error = exc_info.value
            assert "error_crew" in str(error)
            assert "failed during event processing" in str(error)
            assert str(original_exception) in str(error)
            assert error.__cause__ == original_exception
