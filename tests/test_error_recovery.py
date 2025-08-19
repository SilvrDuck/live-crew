"""Comprehensive error recovery tests for production failure scenarios.

These tests validate that live-crew can gracefully handle and recover from
various error conditions that commonly occur in production environments.
They focus on system resilience and ensuring operations continue even when
individual components fail.
"""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock
import asyncio
from typing import Any

import pytest

from live_crew import Orchestrator, Event, Action
from live_crew.config.settings import LiveCrewConfig
from live_crew.crew.handlers import EchoEventHandler
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


class FailingEventHandler:
    """Event handler that fails after specified number of events."""

    def __init__(
        self,
        crew_id: str,
        fail_after: int = 1,
        error_type: type[Exception] = RuntimeError,
    ):
        self.crew_id = crew_id
        self.fail_after = fail_after
        self.error_type = error_type
        self.event_count = 0

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event, failing after specified count."""
        self.event_count += 1
        if self.event_count > self.fail_after:
            raise self.error_type(
                f"Intentional failure from {self.crew_id} after {self.fail_after} events"
            )

        return [
            Action.create(
                f"{self.crew_id}_response",
                {"processed": event.kind, "count": self.event_count},
                stream_id=event.stream_id,
            )
        ]


class IntermittentFailureHandler:
    """Handler that fails intermittently based on event content."""

    def __init__(self, crew_id: str, failure_pattern: list[bool]):
        self.crew_id = crew_id
        self.failure_pattern = failure_pattern  # True = fail, False = succeed
        self.event_index = 0

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with intermittent failures based on pattern."""
        should_fail = self.failure_pattern[self.event_index % len(self.failure_pattern)]
        self.event_index += 1

        if should_fail:
            raise RuntimeError(
                f"Intermittent failure from {self.crew_id} on event {self.event_index}"
            )

        return [
            Action.create(
                f"{self.crew_id}_success",
                {"processed": event.kind, "index": self.event_index},
                stream_id=event.stream_id,
            )
        ]


class SlowEventHandler:
    """Handler that introduces delays to test timeout scenarios."""

    def __init__(self, crew_id: str, delay_seconds: float):
        self.crew_id = crew_id
        self.delay_seconds = delay_seconds

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with artificial delay."""
        await asyncio.sleep(self.delay_seconds)
        return [
            Action.create(
                f"{self.crew_id}_delayed",
                {"processed": event.kind, "delay": self.delay_seconds},
                stream_id=event.stream_id,
            )
        ]


@pytest.fixture
def sample_events():
    """Create sample events for error recovery testing."""
    base_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        EventDict(
            ts=base_time,
            kind="critical_event",
            stream_id="critical",
            payload={"priority": "high", "data": "important"},
        ),
        EventDict(
            ts=base_time + timedelta(milliseconds=50),
            kind="normal_event",
            stream_id="normal",
            payload={"priority": "medium", "data": "standard"},
        ),
        EventDict(
            ts=base_time + timedelta(milliseconds=100),
            kind="batch_event",
            stream_id="batch",
            payload={"priority": "low", "data": "bulk"},
        ),
        EventDict(
            ts=base_time + timedelta(milliseconds=150),
            kind="recovery_event",
            stream_id="recovery",
            payload={"priority": "high", "data": "post_failure"},
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


class TestHandlerFailureRecovery:
    """Test error recovery when individual handlers fail."""

    @pytest.mark.asyncio
    async def test_single_handler_failure_does_not_stop_processing(
        self, temp_events_file
    ):
        """Test that failure in one handler doesn't prevent other handlers from processing."""
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register both failing and successful handlers
            failing_handler = FailingEventHandler("failing_crew", fail_after=1)
            successful_handler = EchoEventHandler("successful_crew")

            scheduler.crew_registry.register_crew(failing_handler, [])
            scheduler.crew_registry.register_crew(successful_handler, [])

            # Process should complete despite failures
            await scheduler.process_events()

            # Successful handler should still process all events (4 events)
            # Failing handler processes 1 event successfully, then fails on subsequent events
            # Total calls should be: 4 (successful) + 1 (failing successful) = 5 minimum
            assert action_transport.publish_action.call_count >= 4

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_handler_failure_types_handled_gracefully(self, temp_events_file):
        """Test that different exception types from handlers are handled gracefully."""
        error_types = [
            ValueError("Invalid data format"),
            TypeError("Type mismatch"),
            KeyError("Missing required field"),
            AttributeError("Object has no attribute"),
            ConnectionError("Network connection failed"),
            TimeoutError("Operation timed out"),
            MemoryError("Out of memory"),
            OSError("System call failed"),
        ]

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register handlers that fail with different exception types
            for i, error_type in enumerate(error_types):
                failing_handler = FailingEventHandler(
                    f"failing_crew_{i}",
                    fail_after=0,  # Fail immediately
                    error_type=type(error_type),
                )
                scheduler.crew_registry.register_crew(failing_handler, [])

            # Add one successful handler to verify processing continues
            successful_handler = EchoEventHandler("successful_crew")
            scheduler.crew_registry.register_crew(successful_handler, [])

            # Should complete without raising any exceptions
            await scheduler.process_events()

            # Successful handler should still process all events
            assert action_transport.publish_action.call_count >= 4

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_intermittent_handler_failures_with_recovery(self, temp_events_file):
        """Test that handlers can recover from intermittent failures."""
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Handler that fails on events 2 and 3, succeeds on 1 and 4
            failure_pattern = [False, True, True, False]  # Success, Fail, Fail, Success
            intermittent_handler = IntermittentFailureHandler(
                "intermittent_crew", failure_pattern
            )

            scheduler.crew_registry.register_crew(intermittent_handler, [])

            await scheduler.process_events()

            # Should have 2 successful actions (events 1 and 4)
            assert action_transport.publish_action.call_count == 2

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_handler_failure_during_context_updates(self, temp_events_file):
        """Test error recovery when handler fails during context backend operations."""

        class FailingContextBackend:
            """Context backend that fails intermittently."""

            def __init__(self):
                self.call_count = 0
                self.contexts: dict[str, dict[int, dict[str, Any]]] = {}

            async def get_snapshot(
                self, stream_id: str, slice_idx: int
            ) -> dict[str, Any]:
                """Get context snapshot (required by protocol)."""
                return self.contexts.get(stream_id, {}).get(slice_idx, {})

            async def apply_diff(
                self, stream_id: str, slice_idx: int, diff: dict[str, Any]
            ) -> None:
                """Apply context diff (required by protocol)."""
                self.call_count += 1
                # Fail on second context update
                if self.call_count == 2:
                    raise RuntimeError("Context backend failure during update")

                if stream_id not in self.contexts:
                    self.contexts[stream_id] = {}
                if slice_idx not in self.contexts[stream_id]:
                    self.contexts[stream_id][slice_idx] = {}
                self.contexts[stream_id][slice_idx].update(diff)

            async def clear_stream(self, stream_id: str) -> None:
                """Clear stream context (required by protocol)."""
                if stream_id in self.contexts:
                    del self.contexts[stream_id]

            def get_context(self, stream_id: str, slice_id: int) -> dict[str, Any]:
                return self.contexts.get(stream_id, {}).get(slice_id, {})

            def get_all_contexts(self) -> dict[str, Any]:
                return self.contexts

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()
            failing_context_backend = FailingContextBackend()

            scheduler = MemoryScheduler(
                config, event_transport, action_transport, failing_context_backend
            )

            # Add handler that will generate context updates
            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            # Should continue processing despite context backend failure
            await scheduler.process_events()

            # All events should still be processed
            assert action_transport.publish_action.call_count == 4

        finally:
            temp_events_file.unlink()


class TestActionTransportFailureRecovery:
    """Test error recovery when action transport fails."""

    @pytest.mark.asyncio
    async def test_action_transport_intermittent_failures(self, temp_events_file):
        """Test recovery when action transport fails intermittently."""

        class FailingActionTransport:
            """Action transport that fails on specific calls."""

            def __init__(self):
                self.call_count = 0
                self.published_actions: list[Action[Any]] = []

            async def publish_action(self, action: Action[Any]) -> None:
                self.call_count += 1
                # Fail on calls 2 and 3
                if self.call_count in [2, 3]:
                    raise ConnectionError(f"Network failure on call {self.call_count}")
                self.published_actions.append(action)

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            failing_action_transport = FailingActionTransport()

            scheduler = MemoryScheduler(
                config, event_transport, failing_action_transport
            )

            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            # Should continue processing despite action transport failures
            await scheduler.process_events()

            # Should have attempted to publish all 4 actions
            assert failing_action_transport.call_count == 4
            # Should have successfully published 2 actions (calls 1 and 4)
            assert len(failing_action_transport.published_actions) == 2

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_action_transport_total_failure_continues_processing(
        self, temp_events_file
    ):
        """Test that total action transport failure doesn't stop event processing."""

        class TotallyFailingActionTransport:
            """Action transport that always fails."""

            def __init__(self):
                self.failure_count = 0

            async def publish_action(self, action: Action[Any]) -> None:
                self.failure_count += 1
                raise ConnectionError(
                    f"Action transport completely down (failure #{self.failure_count})"
                )

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            failing_action_transport = TotallyFailingActionTransport()

            scheduler = MemoryScheduler(
                config, event_transport, failing_action_transport
            )

            echo_handler = EchoEventHandler("test_crew")
            scheduler.crew_registry.register_crew(echo_handler, [])

            # Should complete processing even with total action transport failure
            await scheduler.process_events()

            # Should have attempted to publish all 4 actions
            assert failing_action_transport.failure_count == 4

        finally:
            temp_events_file.unlink()


class TestSystemResourceFailureRecovery:
    """Test error recovery under system resource constraints."""

    @pytest.mark.asyncio
    async def test_memory_pressure_during_processing(self, temp_events_file):
        """Test processing continues under simulated memory pressure."""

        class MemoryPressureHandler:
            """Handler that simulates memory allocation failures."""

            def __init__(self, crew_id: str):
                self.crew_id = crew_id
                self.call_count = 0

            async def handle_event(
                self, event: Event[Any], context: dict[str, Any]
            ) -> list[Action[Any]]:
                self.call_count += 1
                # Simulate memory pressure on every second call
                if self.call_count % 2 == 0:
                    raise MemoryError("Simulated memory allocation failure")

                return [
                    Action.create(
                        f"{self.crew_id}_processed",
                        {"data": event.payload, "call": self.call_count},
                        stream_id=event.stream_id,
                    )
                ]

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            memory_handler = MemoryPressureHandler("memory_crew")
            scheduler.crew_registry.register_crew(memory_handler, [])

            # Should handle memory errors gracefully
            await scheduler.process_events()

            # Should have processed 2 events successfully (calls 1 and 3)
            assert action_transport.publish_action.call_count == 2

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_simulation(self, temp_events_file):
        """Test handling of disk space exhaustion during processing."""

        class DiskSpaceHandler:
            """Handler that simulates disk space issues."""

            def __init__(self, crew_id: str):
                self.crew_id = crew_id
                self.call_count = 0

            async def handle_event(
                self, event: Event[Any], context: dict[str, Any]
            ) -> list[Action[Any]]:
                self.call_count += 1
                # Simulate disk full on third call
                if self.call_count == 3:
                    raise OSError(28, "No space left on device")  # ENOSPC

                return [
                    Action.create(
                        f"{self.crew_id}_saved",
                        {"saved_data": event.payload, "call": self.call_count},
                        stream_id=event.stream_id,
                    )
                ]

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            disk_handler = DiskSpaceHandler("disk_crew")
            scheduler.crew_registry.register_crew(disk_handler, [])

            # Should handle disk space errors gracefully
            await scheduler.process_events()

            # Should have processed 3 events successfully (calls 1, 2, 4)
            assert action_transport.publish_action.call_count == 3

        finally:
            temp_events_file.unlink()


class TestTimeoutAndDeadlockRecovery:
    """Test recovery from timeout and deadlock scenarios."""

    @pytest.mark.asyncio
    async def test_slow_handler_timeout_handling(self, temp_events_file):
        """Test that slow handlers don't block the entire system."""
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Add a very slow handler and a normal handler
            slow_handler = SlowEventHandler("slow_crew", delay_seconds=0.5)
            fast_handler = EchoEventHandler("fast_crew")

            scheduler.crew_registry.register_crew(slow_handler, [])
            scheduler.crew_registry.register_crew(fast_handler, [])

            # Process should complete (slow handlers might be left running)
            await scheduler.process_events()

            # Fast handler should complete quickly
            # Note: In a real implementation, you might want to add timeouts
            # This test validates current behavior doesn't hang indefinitely
            assert (
                action_transport.publish_action.call_count >= 4
            )  # At least fast handler completed

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_concurrent_handler_resource_contention(self, temp_events_file):
        """Test handling of resource contention between handlers."""

        class ResourceContentionHandler:
            """Handler that simulates resource locking contention."""

            _shared_resource_lock = asyncio.Lock()

            def __init__(self, crew_id: str, lock_time: float):
                self.crew_id = crew_id
                self.lock_time = lock_time
                self.processed_count = 0

            async def handle_event(
                self, event: Event[Any], context: dict[str, Any]
            ) -> list[Action[Any]]:
                try:
                    # Simulate acquiring a shared resource with timeout
                    await asyncio.wait_for(
                        ResourceContentionHandler._shared_resource_lock.acquire(),
                        timeout=0.1,
                    )
                    try:
                        # Hold the resource for specified time
                        await asyncio.sleep(self.lock_time)
                        self.processed_count += 1

                        return [
                            Action.create(
                                f"{self.crew_id}_processed",
                                {
                                    "count": self.processed_count,
                                    "lock_time": self.lock_time,
                                },
                                stream_id=event.stream_id,
                            )
                        ]
                    finally:
                        # Always release the lock
                        ResourceContentionHandler._shared_resource_lock.release()

                except asyncio.TimeoutError:
                    # Fail gracefully when can't acquire lock
                    raise RuntimeError(
                        f"Could not acquire resource lock for {self.crew_id}"
                    )

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Add multiple handlers competing for the same resource
            handler1 = ResourceContentionHandler("contention_crew_1", lock_time=0.05)
            handler2 = ResourceContentionHandler("contention_crew_2", lock_time=0.05)

            scheduler.crew_registry.register_crew(handler1, [])
            scheduler.crew_registry.register_crew(handler2, [])

            # Should handle resource contention gracefully
            await scheduler.process_events()

            # Some handlers should succeed, some might fail due to timeouts
            # The system should not deadlock
            assert action_transport.publish_action.call_count >= 1

        finally:
            temp_events_file.unlink()


class TestCompleteSystemFailureRecovery:
    """Test recovery from complete system failure scenarios."""

    @pytest.mark.asyncio
    async def test_cascading_failure_isolation(self, temp_events_file):
        """Test that cascading failures are isolated and don't bring down the system."""

        class CascadingFailureHandler:
            """Handler that can cause cascading failures."""

            failure_registry: dict[str, bool] = {}

            def __init__(self, crew_id: str, triggers_cascade: bool = False):
                self.crew_id = crew_id
                self.triggers_cascade = triggers_cascade

            async def handle_event(
                self, event: Event[Any], context: dict[str, Any]
            ) -> list[Action[Any]]:
                # Check if other handlers have failed and cascade if configured
                if self.triggers_cascade and any(
                    CascadingFailureHandler.failure_registry.values()
                ):
                    CascadingFailureHandler.failure_registry[self.crew_id] = True
                    raise RuntimeError(f"Cascading failure triggered in {self.crew_id}")

                # Simulate random failure for some crews
                if self.crew_id == "unstable_crew":
                    CascadingFailureHandler.failure_registry[self.crew_id] = True
                    raise RuntimeError(f"Initial failure in {self.crew_id}")

                return [
                    Action.create(
                        f"{self.crew_id}_stable",
                        {"processed": event.kind},
                        stream_id=event.stream_id,
                    )
                ]

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(temp_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Add handlers: one that fails, one that cascades, one that's stable
            failing_handler = CascadingFailureHandler(
                "unstable_crew", triggers_cascade=False
            )
            cascading_handler = CascadingFailureHandler(
                "cascade_crew", triggers_cascade=True
            )
            stable_handler = CascadingFailureHandler(
                "stable_crew", triggers_cascade=False
            )

            scheduler.crew_registry.register_crew(failing_handler, [])
            scheduler.crew_registry.register_crew(cascading_handler, [])
            scheduler.crew_registry.register_crew(stable_handler, [])

            # Clear failure registry
            CascadingFailureHandler.failure_registry.clear()

            # Should isolate failures and continue processing
            await scheduler.process_events()

            # Stable handler should process all events despite others failing
            assert action_transport.publish_action.call_count >= 4

        finally:
            temp_events_file.unlink()

    @pytest.mark.asyncio
    async def test_orchestrator_level_failure_recovery(self):
        """Test that orchestrator can recover from critical system failures."""
        # Create events that will trigger various failure modes
        failure_events = [
            {
                "ts": "2025-08-01T10:00:01Z",
                "kind": "system_critical",
                "stream_id": "system",
                "payload": {"action": "trigger_memory_error"},
            },
            {
                "ts": "2025-08-01T10:00:02Z",
                "kind": "network_event",
                "stream_id": "network",
                "payload": {"action": "trigger_connection_error"},
            },
            {
                "ts": "2025-08-01T10:00:03Z",
                "kind": "recovery_event",
                "stream_id": "recovery",
                "payload": {"action": "attempt_recovery"},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(failure_events, f)
            events_file = Path(f.name)

        try:
            # Use orchestrator high-level API
            orchestrator = Orchestrator.from_file(events_file)

            # Register handler that fails based on event content
            class SystemFailureHandler:
                def __init__(self):
                    self.crew_id = "system_failure_handler"

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    payload = event.payload
                    if (
                        isinstance(payload, dict)
                        and payload.get("action") == "trigger_memory_error"
                    ):
                        raise MemoryError("Critical system memory error")
                    elif (
                        isinstance(payload, dict)
                        and payload.get("action") == "trigger_connection_error"
                    ):
                        raise ConnectionError("Network subsystem failure")

                    return [
                        Action.create(
                            "recovery_action",
                            {"recovered_from": event.kind},
                            stream_id=event.stream_id,
                        )
                    ]

            # Register the failing handler
            system_failure_handler = SystemFailureHandler()
            orchestrator.register_handler(system_failure_handler)

            # Should complete despite critical failures
            result = await orchestrator.run()

            # Should have processed all events
            assert result.events_processed == 3
            # Some events may have failed, but processing should complete
            assert result.time_slices >= 1

        finally:
            events_file.unlink()
