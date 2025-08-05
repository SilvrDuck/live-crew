"""Tests for MemoryScheduler integration with all components."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import LiveCrewConfig
from live_crew.crew.handlers import EchoEventHandler
from live_crew.crew.registry import SimpleCrewRegistry
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.console import ConsoleActionTransport
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


@pytest.fixture
def config():
    """Create test configuration."""
    return LiveCrewConfig(slice_ms=100, heartbeat_s=5)


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    base_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        EventDict(
            ts=base_time,
            kind="user_login",
            stream_id="user123",
            payload={"user_id": "123", "ip": "192.168.1.1"},
        ),
        EventDict(
            ts=base_time + timedelta(milliseconds=50),
            kind="page_view",
            stream_id="user123",
            payload={"page": "/dashboard", "user_id": "123"},
        ),
        EventDict(
            ts=base_time + timedelta(milliseconds=150),
            kind="button_click",
            stream_id="user123",
            payload={"button": "save", "user_id": "123"},
        ),
    ]


@pytest.fixture
def temp_events_file(sample_events):
    """Create temporary file with sample events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        events_data = []
        for event in sample_events:
            event_dict = {
                "ts": event.ts.isoformat(),
                "kind": event.kind,
                "stream_id": event.stream_id,
                "payload": event.payload,
            }
            events_data.append(event_dict)

        json.dump(events_data, f, indent=2)
        return Path(f.name)


class TestMemorySchedulerBasics:
    """Test basic MemoryScheduler functionality."""

    def test_scheduler_initialization(self, config):
        """Test scheduler initializes with proper defaults."""
        # Mock validation for dummy file (not testing security here)
        with patch(
            "live_crew.transports.file.validate_file_path",
            return_value=Path("dummy.json"),
        ):
            event_transport = FileEventTransport(Path("dummy.json"))
        scheduler = MemoryScheduler(config, event_transport)

        assert scheduler.config == config
        assert scheduler.event_transport == event_transport
        assert isinstance(scheduler.action_transport, ConsoleActionTransport)
        assert isinstance(scheduler.context_backend, DictContextBackend)
        assert isinstance(scheduler.crew_registry, SimpleCrewRegistry)
        assert scheduler._epoch0 is None
        assert len(scheduler._processed_slices) == 0

    def test_set_epoch(self, config):
        """Test epoch setting."""
        # Mock validation for dummy file (not testing security here)
        with patch(
            "live_crew.transports.file.validate_file_path",
            return_value=Path("dummy.json"),
        ):
            event_transport = FileEventTransport(Path("dummy.json"))
        scheduler = MemoryScheduler(config, event_transport)

        test_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scheduler.set_epoch(test_time)

        assert scheduler._epoch0 == test_time


class TestMemorySchedulerProcessing:
    """Test event processing functionality."""

    @pytest.mark.asyncio
    async def test_process_empty_events(self, config, temp_events_file):
        """Test processing with empty event file."""
        # Create empty events file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            empty_file = Path(f.name)

        try:
            event_transport = FileEventTransport(empty_file)
            scheduler = MemoryScheduler(config, event_transport)

            # Should complete without error
            await scheduler.process_events()
            assert scheduler._epoch0 is None  # No events to set epoch

        finally:
            empty_file.unlink()

    @pytest.mark.asyncio
    async def test_process_events_basic(self, config, temp_events_file):
        """Test basic event processing."""
        try:
            event_transport = FileEventTransport(temp_events_file)

            # Mock action transport to capture output
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register echo handler
            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            await scheduler.process_events()

            # Should have processed 3 events
            assert action_transport.publish_action.call_count == 3

            # Verify epoch was set
            assert scheduler._epoch0 is not None

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_time_slice_grouping(self, config, temp_events_file):
        """Test that events are properly grouped by time slices."""
        try:
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)
            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            await scheduler.process_events()

            # With 100ms slices:
            # Event 1 (t=0): slice 0
            # Event 2 (t=50ms): slice 0
            # Event 3 (t=150ms): slice 1
            # Should process 2 slices
            assert len(scheduler._processed_slices) == 2
            assert 0 in scheduler._processed_slices
            assert 1 in scheduler._processed_slices

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_context_updates(self, config, temp_events_file):
        """Test that context is updated with action metadata."""
        try:
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()
            context_backend = DictContextBackend()

            scheduler = MemoryScheduler(
                config, event_transport, action_transport, context_backend
            )
            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            await scheduler.process_events()

            # Check that context was updated
            contexts = context_backend.get_all_contexts()
            assert "user123" in contexts  # stream_id from events

            # Should have context for both slices
            stream_contexts = contexts["user123"]
            assert 0 in stream_contexts  # slice 0
            assert 1 in stream_contexts  # slice 1

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_multiple_crews(self, config, temp_events_file):
        """Test processing with multiple crews."""
        try:
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register multiple crews
            crew1 = EchoEventHandler("crew_1")
            crew2 = EchoEventHandler("crew_2")
            scheduler.crew_registry.register_crew(crew1, [])
            scheduler.crew_registry.register_crew(crew2, [])

            await scheduler.process_events()

            # Each event should be processed by both crews
            # 3 events * 2 crews = 6 actions
            assert action_transport.publish_action.call_count == 6

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_error_handling(self, config, temp_events_file, capsys):
        """Test error handling when crew processing fails."""
        try:
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create failing handler
            failing_handler = MagicMock()
            failing_handler.crew_id = "failing_crew"
            failing_handler.handle_event = AsyncMock(
                side_effect=Exception("Processing failed")
            )

            scheduler.crew_registry.register_crew(failing_handler, [])

            # Should not raise exception, but continue processing
            await scheduler.process_events()

            # Check error was logged
            captured = capsys.readouterr()
            assert "Error processing event" in captured.out
            assert "failing_crew" in captured.out

        finally:
            temp_events_file.unlink()


class TestMemorySchedulerInterfaces:
    """Test scheduler interface implementation."""

    @pytest.mark.asyncio
    async def test_schedule_crew_interface(self, config):
        """Test schedule_crew interface method."""
        # Mock validation for dummy file (not testing security here)
        with patch(
            "live_crew.transports.file.validate_file_path",
            return_value=Path("dummy.json"),
        ):
            event_transport = FileEventTransport(Path("dummy.json"))
        scheduler = MemoryScheduler(config, event_transport)

        # Should not raise exception (simple implementation)
        await scheduler.schedule_crew("test_crew", 0, [])

    @pytest.mark.asyncio
    async def test_mark_crew_complete_interface(self, config):
        """Test mark_crew_complete interface method."""
        # Mock validation for dummy file (not testing security here)
        with patch(
            "live_crew.transports.file.validate_file_path",
            return_value=Path("dummy.json"),
        ):
            event_transport = FileEventTransport(Path("dummy.json"))
        scheduler = MemoryScheduler(config, event_transport)

        # Should not raise exception (simple implementation)
        await scheduler.mark_crew_complete("test_crew", 0)

    @pytest.mark.asyncio
    async def test_get_pending_crews_interface(self, config):
        """Test get_pending_crews interface method."""
        # Mock validation for dummy file (not testing security here)
        with patch(
            "live_crew.transports.file.validate_file_path",
            return_value=Path("dummy.json"),
        ):
            event_transport = FileEventTransport(Path("dummy.json"))
        scheduler = MemoryScheduler(config, event_transport)

        # Add crews to registry
        crew1 = EchoEventHandler("crew_1")
        crew2 = EchoEventHandler("crew_2")
        scheduler.crew_registry.register_crew(crew1, [])
        scheduler.crew_registry.register_crew(crew2, [])

        pending = await scheduler.get_pending_crews(0)
        assert "crew_1" in pending
        assert "crew_2" in pending
        assert len(pending) == 2


class TestMemorySchedulerConfiguration:
    """Test scheduler with different configurations."""

    @pytest.mark.asyncio
    async def test_custom_slice_duration(self, temp_events_file):
        """Test scheduler with custom slice duration."""
        try:
            # Use larger slice duration
            config = LiveCrewConfig(slice_ms=1000)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)
            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            await scheduler.process_events()

            # With 1000ms slices, all events (0ms, 50ms, 150ms) should be in slice 0
            assert len(scheduler._processed_slices) == 1
            assert 0 in scheduler._processed_slices

        finally:
            temp_events_file.unlink()
