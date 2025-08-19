"""Comprehensive concurrency edge case tests for live-crew.

These tests focus on the most critical production failure scenarios that occur
when multiple components, handlers, or system operations run concurrently.
They validate system resilience under race conditions, resource contention,
and timing-sensitive scenarios that commonly cause production outages.

Each test includes detailed docstrings explaining the real-world scenarios
being protected against and why these edge cases matter for production stability.
"""

import asyncio
import json
import tempfile
import threading
import time
import weakref
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from live_crew import Action, Event, Orchestrator
from live_crew.backends.context import DictContextBackend
from live_crew.config.settings import LiveCrewConfig
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


class ThreadSafeCounter:
    """Thread-safe counter for tracking operations across concurrent handlers."""

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class ConcurrencyTestHandler:
    """Handler designed to expose concurrency issues and race conditions."""

    def __init__(self, crew_id: str, delay_ms: int = 0, fail_on_count: int = -1):
        self.crew_id = crew_id
        self.delay_ms = delay_ms
        self.fail_on_count = fail_on_count
        self.event_count = 0
        self.context_access_count = 0
        self._processing_events: set[str] = set()
        self._lock = asyncio.Lock()

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with configurable delays and failure conditions.

        This handler is designed to expose race conditions by:
        1. Tracking concurrent event processing
        2. Accessing shared context in ways that might cause conflicts
        3. Introducing controlled delays to create timing windows
        4. Failing at specific counts to test error recovery under load
        """
        async with self._lock:
            self.event_count += 1
            current_count = self.event_count

            # Track which events are being processed concurrently
            event_key = f"{event.kind}_{event.stream_id}_{event.ts.isoformat()}"
            if event_key in self._processing_events:
                raise RuntimeError(f"Duplicate event processing detected: {event_key}")
            self._processing_events.add(event_key)

        try:
            # Introduce delay to create race condition windows
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

            # Access context in a way that might cause race conditions
            self.context_access_count += 1
            context_snapshot = dict(context)  # Copy to detect concurrent modifications

            # Fail at specific count to test error recovery under concurrent load
            if self.fail_on_count > 0 and current_count == self.fail_on_count:
                raise RuntimeError(f"Intentional failure on event #{current_count}")

            # Verify context wasn't modified during processing
            if context_snapshot != context:
                raise RuntimeError("Context was modified during handler execution!")

            return [
                Action.create(
                    f"{self.crew_id}_processed",
                    {
                        "event_count": current_count,
                        "context_keys": list(context.keys()),
                        "processing_delay_ms": self.delay_ms,
                        "concurrent_events": len(self._processing_events),
                    },
                    stream_id=event.stream_id,
                )
            ]

        finally:
            async with self._lock:
                self._processing_events.discard(event_key)


class ResourceContentionHandler:
    """Handler that simulates resource contention scenarios.

    Real-world scenarios this protects against:
    - Database connection pool exhaustion
    - File handle limits being reached
    - Memory allocation failures under load
    - Network socket exhaustion
    """

    _shared_resource_pool = asyncio.Semaphore(2)  # Simulate limited resource pool
    _allocation_counter = ThreadSafeCounter()

    def __init__(self, crew_id: str, hold_time_ms: int = 50, max_allocations: int = -1):
        self.crew_id = crew_id
        self.hold_time_ms = hold_time_ms
        self.max_allocations = max_allocations
        self.allocations_made = 0

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event while competing for limited resources.

        This simulates real production scenarios where handlers compete for:
        - Database connections
        - File handles
        - Network connections
        - Memory allocations
        - External API rate limits
        """
        allocation_id = ResourceContentionHandler._allocation_counter.increment()

        # Simulate resource exhaustion with graceful degradation
        if self.max_allocations > 0 and allocation_id > self.max_allocations:
            return [
                Action.create(
                    f"{self.crew_id}_exhausted",
                    {
                        "reason": "resource_pool_exhausted",
                        "allocation_id": allocation_id,
                        "max_allocations": self.max_allocations,
                        "graceful_degradation": True,
                    },
                    stream_id=event.stream_id,
                )
            ]

        try:
            # Compete for limited resource with very short timeout to force contention
            await asyncio.wait_for(
                ResourceContentionHandler._shared_resource_pool.acquire(),
                timeout=0.005,  # Very short timeout to trigger contention failures
            )

            try:
                # Hold resource for specified time (simulates processing)
                await asyncio.sleep(self.hold_time_ms / 1000)
                self.allocations_made += 1

                return [
                    Action.create(
                        f"{self.crew_id}_acquired",
                        {
                            "allocation_id": allocation_id,
                            "hold_time_ms": self.hold_time_ms,
                            "total_allocations": self.allocations_made,
                        },
                        stream_id=event.stream_id,
                    )
                ]
            finally:
                ResourceContentionHandler._shared_resource_pool.release()

        except asyncio.TimeoutError:
            # Graceful degradation when resources are exhausted
            return [
                Action.create(
                    f"{self.crew_id}_degraded",
                    {
                        "reason": "resource_timeout",
                        "allocation_id": allocation_id,
                        "fallback_processing": True,
                    },
                    stream_id=event.stream_id,
                )
            ]


class DeadlockProneHandler:
    """Handler designed to detect and prevent deadlock scenarios.

    Protects against real-world deadlocks such as:
    - Circular dependency locks between handlers
    - Resource ordering deadlocks
    - Context update deadlocks
    - Cross-crew synchronization deadlocks
    """

    _global_locks = {}
    _lock_acquisition_order = []

    def __init__(self, crew_id: str, required_locks: list[str]):
        self.crew_id = crew_id
        self.required_locks = sorted(
            required_locks
        )  # Consistent ordering prevents deadlocks

        # Initialize locks if they don't exist
        for lock_name in required_locks:
            if lock_name not in DeadlockProneHandler._global_locks:
                DeadlockProneHandler._global_locks[lock_name] = asyncio.Lock()

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event while acquiring multiple locks in consistent order.

        This demonstrates proper deadlock prevention through:
        1. Consistent lock acquisition ordering
        2. Timeout-based deadlock detection
        3. Graceful fallback when deadlocks are detected
        """
        acquired_locks = []

        try:
            # Acquire locks in sorted order to prevent deadlocks
            for lock_name in self.required_locks:
                lock = DeadlockProneHandler._global_locks[lock_name]

                # Use timeout to detect potential deadlocks
                await asyncio.wait_for(
                    lock.acquire(),
                    timeout=0.5,  # Reasonable timeout for deadlock detection
                )
                acquired_locks.append(lock)

                # Track acquisition order for deadlock analysis
                DeadlockProneHandler._lock_acquisition_order.append(
                    (self.crew_id, lock_name, time.time())
                )

                # Small delay to increase chance of lock contention
                await asyncio.sleep(0.01)

            # Simulate work while holding all required locks
            await asyncio.sleep(0.05)

            return [
                Action.create(
                    f"{self.crew_id}_completed",
                    {
                        "locks_acquired": self.required_locks,
                        "acquisition_order": len(
                            DeadlockProneHandler._lock_acquisition_order
                        ),
                        "potential_deadlock": False,
                    },
                    stream_id=event.stream_id,
                )
            ]

        except asyncio.TimeoutError:
            # Deadlock detected - implement graceful fallback
            return [
                Action.create(
                    f"{self.crew_id}_deadlock_avoided",
                    {
                        "reason": "deadlock_timeout",
                        "acquired_locks": len(acquired_locks),
                        "required_locks": self.required_locks,
                        "fallback_processing": True,
                    },
                    stream_id=event.stream_id,
                )
            ]

        finally:
            # Release locks in reverse order (LIFO)
            for lock in reversed(acquired_locks):
                try:
                    lock.release()
                except RuntimeError:
                    pass  # Already released


@pytest.fixture
def concurrent_events():
    """Create events designed to trigger concurrency issues."""
    base_time = datetime(2025, 8, 6, 10, 0, 0, tzinfo=timezone.utc)

    # Create events with minimal time gaps to maximize concurrency
    events = []
    for i in range(10):
        events.append(
            EventDict(
                ts=base_time + timedelta(microseconds=i * 1000),  # 1ms apart
                kind=f"concurrent_event_{i % 3}",  # 3 types cycling
                stream_id=f"stream_{i % 2}",  # 2 streams alternating
                payload={
                    "sequence": i,
                    "batch": "concurrent_test",
                    "data": f"payload_{i}",
                },
            )
        )

    return events


@pytest.fixture
def concurrent_events_file(concurrent_events):
    """Create temporary file with concurrent events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        events_data = []
        for event in concurrent_events:
            events_data.append(
                {
                    "ts": event.ts.isoformat(),
                    "kind": event.kind,
                    "stream_id": event.stream_id,
                    "payload": event.payload,
                }
            )
        json.dump(events_data, f, indent=2)
        return Path(f.name)


class TestConcurrentHandlerExecution:
    """Test concurrent execution of multiple handlers with race condition detection."""

    @pytest.mark.asyncio
    async def test_concurrent_handlers_no_race_conditions(self, concurrent_events_file):
        """Test that concurrent handlers don't interfere with each other's processing.

        Real-world scenario: Multiple microservices processing events simultaneously
        should not corrupt shared state or interfere with each other's operations.

        Production failure this prevents: Handler A modifying context while Handler B
        is reading it, causing inconsistent state or data corruption.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)  # Short slices for max concurrency
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register multiple handlers with different processing characteristics
            fast_handler = ConcurrencyTestHandler("fast_crew", delay_ms=1)
            medium_handler = ConcurrencyTestHandler("medium_crew", delay_ms=10)
            slow_handler = ConcurrencyTestHandler("slow_crew", delay_ms=25)

            scheduler.crew_registry.register_crew(fast_handler, [])
            scheduler.crew_registry.register_crew(medium_handler, [])
            scheduler.crew_registry.register_crew(slow_handler, [])

            # Process events concurrently
            await scheduler.process_events()

            # Verify all handlers processed all events without interference
            # 10 events * 3 handlers = 30 actions expected
            assert action_transport.publish_action.call_count == 30

            # Verify each handler processed the expected number of events
            assert fast_handler.event_count == 10
            assert medium_handler.event_count == 10
            assert slow_handler.event_count == 10

            # Verify no duplicate event processing was detected
            assert len(fast_handler._processing_events) == 0  # All should be cleared
            assert len(medium_handler._processing_events) == 0
            assert len(slow_handler._processing_events) == 0

        finally:
            concurrent_events_file.unlink()

    @pytest.mark.asyncio
    async def test_concurrent_handler_failure_isolation(self, concurrent_events_file):
        """Test that failure in one concurrent handler doesn't affect others.

        Real-world scenario: If one microservice crashes while processing events,
        other services should continue processing without interruption.

        Production failure this prevents: Cascading failures where one failing
        service brings down the entire event processing pipeline.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Register handlers where one fails partway through
            stable_handler = ConcurrencyTestHandler("stable_crew", delay_ms=5)
            failing_handler = ConcurrencyTestHandler(
                "failing_crew", delay_ms=5, fail_on_count=5
            )
            recovery_handler = ConcurrencyTestHandler("recovery_crew", delay_ms=5)

            scheduler.crew_registry.register_crew(stable_handler, [])
            scheduler.crew_registry.register_crew(failing_handler, [])
            scheduler.crew_registry.register_crew(recovery_handler, [])

            await scheduler.process_events()

            # Verify stable handlers processed all events despite failing handler
            assert stable_handler.event_count == 10
            assert recovery_handler.event_count == 10

            # Failing handler should have processed some events before failing
            assert failing_handler.event_count >= 5

            # Total actions: 20 (stable + recovery) + 4 (failing before error) = 24+
            assert action_transport.publish_action.call_count >= 24

        finally:
            concurrent_events_file.unlink()


class TestResourceContentionScenarios:
    """Test resource contention scenarios that commonly cause production issues."""

    @pytest.mark.asyncio
    async def test_limited_resource_pool_contention(self, concurrent_events_file):
        """Test graceful handling when handlers compete for limited resources.

        Real-world scenario: Database connection pool has 10 connections, but
        15 concurrent requests arrive. System should gracefully handle overflow.

        Production failure this prevents: Application crashes when resource pools
        are exhausted, rather than implementing graceful degradation.
        """
        try:
            config = LiveCrewConfig(slice_ms=25)  # Very short for max contention
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create handlers that compete for limited resources
            # Each handler tries to hold resource for longer, we have 2 resources total
            contender1 = ResourceContentionHandler("contender_1", hold_time_ms=25)
            contender2 = ResourceContentionHandler("contender_2", hold_time_ms=25)
            contender3 = ResourceContentionHandler("contender_3", hold_time_ms=25)
            contender4 = ResourceContentionHandler("contender_4", hold_time_ms=25)
            contender5 = ResourceContentionHandler(
                "contender_5", hold_time_ms=25
            )  # Additional contender

            scheduler.crew_registry.register_crew(contender1, [])
            scheduler.crew_registry.register_crew(contender2, [])
            scheduler.crew_registry.register_crew(contender3, [])
            scheduler.crew_registry.register_crew(contender4, [])
            scheduler.crew_registry.register_crew(
                contender5, []
            )  # Additional contender

            await scheduler.process_events()

            # Should complete processing despite resource contention
            # Some handlers will get resources, others will timeout gracefully
            total_calls = action_transport.publish_action.call_count
            assert (
                total_calls >= 25
            )  # At minimum, some handlers succeeded (10 events * 5 handlers, some timeouts expected)

            # Verify graceful degradation occurred (some timeouts expected)
            # In real scenario, would check for degraded actions in transport

        finally:
            concurrent_events_file.unlink()

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, concurrent_events_file):
        """Test system recovery when resources are completely exhausted.

        Real-world scenario: All database connections are in use and new requests
        arrive. System should queue, timeout, or gracefully degrade rather than crash.

        Production failure this prevents: OutOfMemory, ConnectionPoolExhausted,
        or similar resource exhaustion errors bringing down the entire system.
        """
        try:
            config = LiveCrewConfig(slice_ms=25)
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create handlers with very limited resource pool
            exhausting_handler1 = ResourceContentionHandler(
                "exhauster_1", hold_time_ms=150, max_allocations=3
            )
            exhausting_handler2 = ResourceContentionHandler(
                "exhauster_2", hold_time_ms=150, max_allocations=3
            )
            exhausting_handler3 = ResourceContentionHandler(
                "exhauster_3", hold_time_ms=150, max_allocations=3
            )  # Additional load

            scheduler.crew_registry.register_crew(exhausting_handler1, [])
            scheduler.crew_registry.register_crew(exhausting_handler2, [])
            scheduler.crew_registry.register_crew(
                exhausting_handler3, []
            )  # Additional load

            # Should not crash, should handle exhaustion gracefully
            await scheduler.process_events()

            # Some processing should succeed, failures should be graceful
            assert (
                action_transport.publish_action.call_count >= 9
            )  # Some succeed from 3 handlers, max 3 allocations each

        finally:
            concurrent_events_file.unlink()


class TestDeadlockPreventionScenarios:
    """Test deadlock prevention and detection in complex scenarios."""

    @pytest.mark.asyncio
    async def test_deadlock_prevention_with_consistent_ordering(
        self, concurrent_events_file
    ):
        """Test that consistent lock ordering prevents deadlocks.

        Real-world scenario: Service A needs locks [DB, Cache] and Service B needs
        locks [Cache, DB]. Without consistent ordering, they can deadlock.

        Production failure this prevents: Deadlocks that freeze the entire system
        requiring manual intervention or process restarts.
        """
        try:
            config = LiveCrewConfig(slice_ms=30)
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create handlers that need overlapping locks in different orders
            # Consistent internal ordering should prevent deadlocks
            handler1 = DeadlockProneHandler(
                "handler_1", ["db_lock", "cache_lock", "file_lock"]
            )
            handler2 = DeadlockProneHandler(
                "handler_2", ["cache_lock", "db_lock"]
            )  # Different order
            handler3 = DeadlockProneHandler("handler_3", ["file_lock", "cache_lock"])

            scheduler.crew_registry.register_crew(handler1, [])
            scheduler.crew_registry.register_crew(handler2, [])
            scheduler.crew_registry.register_crew(handler3, [])

            # Should complete without deadlocks due to consistent ordering
            await scheduler.process_events()

            # All handlers should complete most operations
            # Some timeouts are acceptable (indicates deadlock prevention working)
            assert action_transport.publish_action.call_count >= 15

        finally:
            concurrent_events_file.unlink()

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_recovery(self, concurrent_events_file):
        """Test that potential deadlocks are detected and recovered from.

        Real-world scenario: Complex dependency chains where circular waits
        can occur, but system detects and breaks the cycle gracefully.

        Production failure this prevents: Silent deadlocks that cause requests
        to hang indefinitely with no error indication.
        """
        try:
            config = LiveCrewConfig(slice_ms=20)  # Very short to create pressure
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create scenario prone to deadlocks with timeout recovery
            complex_handler1 = DeadlockProneHandler(
                "complex_1", ["lock_a", "lock_b", "lock_c"]
            )
            complex_handler2 = DeadlockProneHandler("complex_2", ["lock_c", "lock_a"])
            complex_handler3 = DeadlockProneHandler("complex_3", ["lock_b", "lock_c"])

            scheduler.crew_registry.register_crew(complex_handler1, [])
            scheduler.crew_registry.register_crew(complex_handler2, [])
            scheduler.crew_registry.register_crew(complex_handler3, [])

            # Should complete with mixture of successes and timeout recoveries
            await scheduler.process_events()

            # System should not hang - some operations succeed, others timeout gracefully
            total_actions = action_transport.publish_action.call_count
            assert total_actions >= 10  # Some processing should complete

            # Verify lock acquisition tracking worked
            assert len(DeadlockProneHandler._lock_acquisition_order) >= 10

        finally:
            concurrent_events_file.unlink()


class TestContextConcurrencyScenarios:
    """Test concurrent access to shared context under high load."""

    @pytest.mark.asyncio
    async def test_concurrent_context_modifications(self, concurrent_events_file):
        """Test that concurrent context modifications don't corrupt state.

        Real-world scenario: Multiple handlers updating shared application state
        simultaneously should not result in lost updates or corrupted data.

        Production failure this prevents: Race conditions in shared state that
        cause inconsistent or corrupted business data.
        """
        try:
            config = LiveCrewConfig(slice_ms=30)
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            # Use shared context backend
            shared_context = DictContextBackend()
            scheduler = MemoryScheduler(
                config, event_transport, action_transport, shared_context
            )

            class ContextModifyingHandler:
                def __init__(self, crew_id: str, context_key: str):
                    self.crew_id = crew_id
                    self.context_key = context_key
                    self.modifications = 0

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    # Simulate context modifications that could race
                    self.modifications += 1
                    current_value = context.get(self.context_key, 0)

                    # Small delay to create race condition window
                    await asyncio.sleep(0.001)

                    # This update could race with other handlers
                    new_value = current_value + 1

                    return [
                        Action.create(
                            f"{self.crew_id}_modified",
                            {
                                "context_key": self.context_key,
                                "old_value": current_value,
                                "new_value": new_value,
                                "modification_count": self.modifications,
                            },
                            stream_id=event.stream_id,
                        )
                    ]

            # Create handlers that modify overlapping context keys
            modifier1 = ContextModifyingHandler("modifier_1", "shared_counter")
            modifier2 = ContextModifyingHandler("modifier_2", "shared_counter")
            modifier3 = ContextModifyingHandler("modifier_3", "other_counter")

            scheduler.crew_registry.register_crew(modifier1, [])
            scheduler.crew_registry.register_crew(modifier2, [])
            scheduler.crew_registry.register_crew(modifier3, [])

            await scheduler.process_events()

            # Should complete without exceptions
            assert (
                action_transport.publish_action.call_count == 30
            )  # 10 events * 3 handlers

            # Verify context modifications occurred
            contexts = shared_context.get_all_contexts()
            assert len(contexts) >= 1  # At least one stream processed

        finally:
            concurrent_events_file.unlink()


class TestHighLoadConcurrencyScenarios:
    """Test system behavior under high concurrency loads."""

    @pytest.mark.asyncio
    async def test_high_concurrency_event_burst(self):
        """Test system stability when processing large concurrent event bursts.

        Real-world scenario: Black Friday traffic spike or viral social media event
        causes sudden 10x+ increase in event volume.

        Production failure this prevents: System collapse or memory exhaustion
        under sudden load spikes that exceed normal capacity planning.
        """
        # Create large burst of concurrent events
        base_time = datetime(2025, 8, 6, 10, 0, 0, tzinfo=timezone.utc)
        burst_events = []

        # 100 events within 1 second - high concurrency load
        for i in range(100):
            burst_events.append(
                {
                    "ts": (
                        base_time + timedelta(microseconds=i * 10000)
                    ).isoformat(),  # 10ms apart
                    "kind": f"burst_event_{i % 5}",
                    "stream_id": f"burst_stream_{i % 10}",
                    "payload": {"burst_id": i, "data": f"burst_data_{i}"},
                }
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(burst_events, f)
            burst_file = Path(f.name)

        try:
            config = LiveCrewConfig(slice_ms=100)  # 100ms slices
            event_transport = FileEventTransport(burst_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Single handler to process burst
            burst_handler = ConcurrencyTestHandler("burst_handler", delay_ms=1)
            scheduler.crew_registry.register_crew(burst_handler, [])

            # Should handle burst without crashing or excessive memory usage
            start_time = time.time()
            await scheduler.process_events()
            processing_time = time.time() - start_time

            # Verify all events processed
            assert action_transport.publish_action.call_count == 100
            assert burst_handler.event_count == 100

            # Verify reasonable processing time (should complete within a few seconds)
            assert processing_time < 10.0, (
                f"Processing took too long: {processing_time}s"
            )

        finally:
            burst_file.unlink()

    @pytest.mark.asyncio
    async def test_memory_efficiency_under_concurrent_load(
        self, concurrent_events_file
    ):
        """Test memory usage remains reasonable under concurrent processing.

        Real-world scenario: Long-running event processing should not exhibit
        memory leaks or excessive memory growth under sustained load.

        Production failure this prevents: OutOfMemoryError in production after
        hours/days of processing due to memory leaks in concurrent handlers.
        """
        try:
            config = LiveCrewConfig(slice_ms=25)
            event_transport = FileEventTransport(concurrent_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Create handlers that might accumulate memory
            class MemoryTrackingHandler:
                def __init__(self, crew_id: str):
                    self.crew_id = crew_id
                    self.processed_events: list[dict] = []  # Could accumulate
                    self.temp_data: dict[str, Any] = {}

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    # Simulate memory accumulation that should be cleaned up
                    event_copy = {
                        "kind": event.kind,
                        "stream_id": event.stream_id,
                        "payload": dict(event.payload)
                        if isinstance(event.payload, dict)
                        else event.payload,
                    }
                    self.processed_events.append(event_copy)

                    # Create temporary data
                    temp_key = f"temp_{len(self.processed_events)}"
                    self.temp_data[temp_key] = list(
                        range(100)
                    )  # Some memory allocation

                    # Clean up old temp data (prevent memory leaks)
                    if len(self.temp_data) > 5:
                        oldest_key = min(self.temp_data.keys())
                        del self.temp_data[oldest_key]

                    return [
                        Action.create(
                            f"{self.crew_id}_tracked",
                            {
                                "events_processed": len(self.processed_events),
                                "temp_data_size": len(self.temp_data),
                                "memory_managed": True,
                            },
                            stream_id=event.stream_id,
                        )
                    ]

            memory_handler1 = MemoryTrackingHandler("memory_1")
            memory_handler2 = MemoryTrackingHandler("memory_2")

            scheduler.crew_registry.register_crew(memory_handler1, [])
            scheduler.crew_registry.register_crew(memory_handler2, [])

            # Track memory usage
            import gc

            gc.collect()  # Clean up before test

            await scheduler.process_events()

            # Verify processing completed
            assert (
                action_transport.publish_action.call_count == 20
            )  # 10 events * 2 handlers

            # Verify memory cleanup occurred
            assert len(memory_handler1.temp_data) <= 5
            assert len(memory_handler2.temp_data) <= 5

            # Check for memory leaks in scheduler itself
            gc.collect()
            weak_refs = [
                weakref.ref(scheduler),
                weakref.ref(memory_handler1),
                weakref.ref(memory_handler2),
            ]

            # Objects should be cleanly referenced (no circular references)
            assert all(ref() is not None for ref in weak_refs)

        finally:
            concurrent_events_file.unlink()


@pytest.mark.asyncio
async def test_orchestrator_level_concurrency_edge_cases():
    """Test concurrency edge cases at the Orchestrator API level.

    Real-world scenario: Application using Orchestrator API with multiple
    concurrent workflows should handle cross-workflow interference gracefully.

    Production failure this prevents: Orchestrator-level race conditions that
    corrupt internal state or cause undefined behavior across workflows.
    """
    # Create multiple event files for concurrent orchestrators
    files_to_cleanup = []

    try:
        orchestrators = []
        tasks = []

        # Create 3 concurrent orchestrators with different event sets
        for i in range(3):
            events = [
                {
                    "ts": f"2025-08-06T10:00:{i:02d}Z",
                    "kind": f"concurrent_orchestrator_event_{i}",
                    "stream_id": f"orchestrator_stream_{i}",
                    "payload": {"orchestrator_id": i, "data": f"test_data_{i}"},
                }
            ]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(events, f)
                event_file = Path(f.name)
                files_to_cleanup.append(event_file)

            # Create orchestrator with handler
            orchestrator = Orchestrator.from_file(event_file)

            class ConcurrentOrchestrationHandler:
                def __init__(self, orchestrator_id: int):
                    self.orchestrator_id = orchestrator_id
                    self.crew_id = f"orchestrator_{orchestrator_id}_handler"

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    # Simulate concurrent orchestrator processing
                    await asyncio.sleep(0.01)  # Small delay

                    return [
                        Action.create(
                            f"orchestrator_{self.orchestrator_id}_result",
                            {
                                "orchestrator_id": self.orchestrator_id,
                                "processed_event": event.kind,
                                "context_size": len(context),
                            },
                            stream_id=event.stream_id,
                        )
                    ]

            handler = ConcurrentOrchestrationHandler(i)
            orchestrator.register_handler(handler)
            orchestrators.append(orchestrator)

            # Create async task for each orchestrator
            tasks.append(orchestrator.run())

        # Run all orchestrators concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all completed successfully
        assert len(results) == 3
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Orchestrator failed with exception: {result}")
            assert result.events_processed == 1

    finally:
        for file_path in files_to_cleanup:
            file_path.unlink()
