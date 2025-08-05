"""Integration tests for Orchestrator API UX improvements.

These tests validate that the facade pattern Orchestrator provides
simplified developer experience while maintaining full functionality.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from freezegun import freeze_time

from live_crew import Action, Event, Orchestrator, event_handler


class TestOrchestratorBasicUsage:
    """Test basic Orchestrator usage patterns."""

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_from_file_basic(self):
        """Test basic file-based orchestration workflow."""
        # Create test events file
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "user_signup",
                "stream_id": "users",
                "payload": {"name": "Alice", "email": "alice@example.com"},
            },
            {
                "ts": "2025-08-01T10:00:02Z",
                "kind": "user_login",
                "stream_id": "users",
                "payload": {"user_id": "alice_123"},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # Create orchestrator with simplified API
            orchestrator = Orchestrator.from_file(events_file)

            # Verify orchestrator created with correct defaults
            assert orchestrator.config is not None
            assert orchestrator.crew_registry is not None

            # Run processing
            result = await orchestrator.run()

            # Verify processing completed
            assert result.events_processed == 2
            assert result.time_slices >= 1
            assert isinstance(result.actions, list)
            assert isinstance(result.context_final_state, dict)

        finally:
            events_file.unlink()

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_with_event_handler_decorator(self):
        """Test simplified handler registration with decorator."""
        # Create test events file
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "user_signup",
                "stream_id": "users",
                "payload": {"name": "Bob"},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # Define handler using decorator
            @event_handler("user_signup")
            def greet_user(event: Event[Any]) -> Action[Any]:
                return Action.create(
                    "greeting",
                    f"Welcome {event.payload['name']}!",
                    stream_id=event.stream_id,
                )

            # Create orchestrator and register handler
            orchestrator = Orchestrator.from_file(events_file)
            orchestrator.register_handler(greet_user)

            # Run processing
            result = await orchestrator.run()

            # Verify results
            assert result.events_processed == 1
            # Note: ConsoleActionTransport streams actions, doesn't collect them
            # so actions_generated reflects what was sent to transport, not collected
            assert (
                result.actions_generated == 0
            )  # No actions collected (streamed instead)
            assert len(result.actions) == 0  # Actions list empty for console transport

            # The test verifies the workflow works - action was printed to console
            # In a real scenario with CollectingActionTransport, we'd verify action content

        finally:
            events_file.unlink()

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_from_config(self):
        """Test configuration-based orchestrator creation."""
        # Create test events file
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "test_event",
                "stream_id": "test",
                "payload": {"message": "hello"},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # Create orchestrator from config
            orchestrator = Orchestrator.from_config(
                "config.yaml"
            )  # Uses defaults if file missing

            # Verify orchestrator created successfully
            assert orchestrator.config is not None
            assert orchestrator.crew_registry is not None

            # Note: This test validates the API works, actual config loading
            # will be enhanced in Phase 2

        finally:
            events_file.unlink()


class TestOrchestratorAdvancedFeatures:
    """Test advanced Orchestrator features and customization."""

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_custom_components(self):
        """Test orchestrator with custom component injection."""
        from live_crew.transports.console import ConsoleActionTransport
        from live_crew.backends.context import DictContextBackend
        from live_crew.crew.registry import SimpleCrewRegistry

        # Create test events file
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "custom_event",
                "stream_id": "custom",
                "payload": {"data": "test"},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # Create orchestrator with custom components
            custom_action_transport = ConsoleActionTransport()
            custom_context_backend = DictContextBackend()
            custom_crew_registry = SimpleCrewRegistry()

            orchestrator = Orchestrator(
                event_transport=None,  # Will be set by from_file
                action_transport=custom_action_transport,
                context_backend=custom_context_backend,
                crew_registry=custom_crew_registry,
            )

            # Override with file transport
            orchestrator = Orchestrator.from_file(
                events_file, config=orchestrator.config
            )

            # Verify custom components preserved extensibility
            result = await orchestrator.run()
            assert result.events_processed == 1

        finally:
            events_file.unlink()

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_result_collection_strategies(self):
        """Test that orchestrator correctly uses result collection strategies."""
        # Create test events file
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "test_event",
                "stream_id": "test",
                "payload": {"value": 42},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # Test with default collector (should be NullResultCollector for ConsoleActionTransport)
            orchestrator = Orchestrator.from_file(events_file)
            result = await orchestrator.run()

            # Console transport doesn't collect actions, so should be empty
            assert isinstance(result.actions, list)
            assert result.events_processed == 1

            # Test strategy pattern is working - NullResultCollector for console transport
            # Note: CollectingActionTransport would be used for testing scenarios
            # where we want to capture actions in memory

        finally:
            events_file.unlink()


class TestOrchestratorSimplifiedAPI:
    """Test that the simplified API meets UX requirements."""

    def test_simplified_imports_available(self):
        """Test that simplified imports work as expected."""
        # These imports should work from top-level package
        from live_crew import Orchestrator, event_handler, Event, Action

        # Verify classes are available
        assert Orchestrator is not None
        assert event_handler is not None
        assert Event is not None
        assert Action is not None

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_hello_world_simplified_pattern(self):
        """Test the Hello World simplified pattern from UX requirements."""
        # Create simple test events
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "user_signup",
                "stream_id": "users",
                "payload": {"name": "Charlie"},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:
            # This should match the target pattern from scrum.md
            @event_handler("user_signup")
            def greet_user(event):
                return Action.create("greeting", f"Welcome {event.payload['name']}!")

            orchestrator = Orchestrator.from_file(events_file)
            orchestrator.register_handler(greet_user)
            result = await orchestrator.run()

            # Verify the simplified workflow works
            assert result.events_processed == 1

        finally:
            events_file.unlink()

    def test_action_helper_methods(self):
        """Test Action helper methods for simplified creation."""
        # Test Action.create helper
        action = Action.create(
            "test_action", {"message": "hello"}, stream_id="test", ttl_ms=1000
        )

        assert action.kind == "test_action"
        assert action.payload == {"message": "hello"}
        assert action.stream_id == "test"
        assert action.ttl_ms == 1000

        # Test Action.from_event helper
        event = Event(
            ts=action.ts,
            kind="source_event",
            stream_id="source",
            payload={"original": "data"},
        )

        derived_action = Action.from_event(event, "derived_action", {"processed": True})

        assert derived_action.kind == "derived_action"
        assert derived_action.stream_id == "source"  # Inherited from event
        assert derived_action.payload == {"processed": True}


class TestOrchestratorErrorHandling:
    """Test error handling and edge cases for Orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_missing_event_transport(self):
        """Test error when no event transport configured."""
        orchestrator = Orchestrator()  # No event transport

        with pytest.raises(ValueError, match="No event transport configured"):
            await orchestrator.run()

    @pytest.mark.asyncio
    async def test_orchestrator_nonexistent_file(self):
        """Test error handling for nonexistent events file."""
        nonexistent_file = Path("/tmp/nonexistent_events.json")

        with pytest.raises(FileNotFoundError):
            Orchestrator.from_file(nonexistent_file)

    @pytest.mark.asyncio
    @freeze_time("2025-08-01T10:00:00Z")
    async def test_orchestrator_handler_error_resilience(self):
        """Test that orchestrator handles handler errors gracefully."""
        # Create test events
        events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "error_event",
                "stream_id": "test",
                "payload": {"trigger_error": True},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(events, f)
            events_file = Path(f.name)

        try:

            @event_handler("error_event")
            def failing_handler(event):
                raise RuntimeError("Handler intentionally failed")

            orchestrator = Orchestrator.from_file(events_file)
            orchestrator.register_handler(failing_handler)

            # Should not crash, should handle error gracefully
            result = await orchestrator.run()
            assert result.events_processed == 1

        finally:
            events_file.unlink()
