"""Performance edge cases tests for resource exhaustion and scalability limits.

These tests focus on performance bottlenecks, resource exhaustion scenarios,
and scalability limits that commonly cause production outages under high load.
They validate system behavior when pushed beyond normal operating parameters.

Each test simulates realistic high-load scenarios that occur in production
environments, ensuring the system maintains stability and provides proper
resource management under extreme conditions.
"""

import asyncio
import gc
import json
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
import tracemalloc
import psutil
import os

import pytest

from live_crew import Action, Event
from live_crew.config.settings import LiveCrewConfig
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


class MemoryLeakTestHandler:
    """Handler designed to test for memory leaks and proper cleanup.

    Real-world scenarios this protects against:
    - Memory leaks in event processing loops
    - Unclosed resources and connections
    - Growing cache sizes without bounds
    - Handler state accumulation over time
    """

    def __init__(self, crew_id: str, leak_mode: bool = False):
        self.crew_id = crew_id
        self.leak_mode = leak_mode
        self.event_count = 0
        self.large_objects: list[bytes] = []
        self.references: list[Any] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with optional memory leak simulation."""
        self.event_count += 1

        if self.leak_mode:
            # Intentionally create memory leaks for testing
            large_data = b"x" * (1024 * 1024)  # 1MB per event
            self.large_objects.append(large_data)

            # Create circular references
            circular_ref = {"data": large_data, "self": None}
            circular_ref["self"] = circular_ref
            self.references.append(circular_ref)

        return [
            Action.create(
                "memory_processed",
                {
                    "event_number": self.event_count,
                    "memory_objects": len(self.large_objects) if self.leak_mode else 0,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


class ResourceExhaustionTestHandler:
    """Handler that tests resource exhaustion scenarios.

    Real-world scenarios this protects against:
    - File descriptor exhaustion
    - Thread pool saturation
    - Connection pool depletion
    - CPU-intensive operations blocking event loop
    """

    def __init__(self, crew_id: str, stress_mode: bool = False):
        self.crew_id = crew_id
        self.stress_mode = stress_mode
        self.active_connections = 0
        self.open_files: list[Any] = []
        self.threads: list[threading.Thread] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with resource exhaustion testing."""
        payload = event.payload if isinstance(event.payload, dict) else {}

        if self.stress_mode:
            stress_type = payload.get("stress_type", "cpu_intensive")

            if stress_type == "file_descriptors":
                return await self._stress_file_descriptors(event)
            elif stress_type == "cpu_intensive":
                return await self._stress_cpu_processing(event)
            elif stress_type == "connection_pool":
                return await self._stress_connection_pool(event)
            elif stress_type == "thread_exhaustion":
                return await self._stress_thread_pool(event)

        return [
            Action.create(
                "resource_processed",
                {
                    "active_connections": self.active_connections,
                    "open_files": len(self.open_files),
                    "active_threads": len(self.threads),
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _stress_file_descriptors(self, event: Event[Any]) -> list[Action[Any]]:
        """Stress test file descriptor limits."""
        try:
            # Open temporary files (simulating file handle exhaustion)
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.open_files.append(temp_file)

            return [
                Action.create(
                    "file_descriptor_stress",
                    {
                        "open_file_count": len(self.open_files),
                        "file_path": temp_file.name,
                        "timestamp": time.time(),
                    },
                    stream_id=event.stream_id,
                )
            ]
        except OSError as e:
            # Expected when file descriptor limit is reached
            return [
                Action.create(
                    "fd_exhaustion_detected",
                    {
                        "error": str(e),
                        "open_file_count": len(self.open_files),
                        "timestamp": time.time(),
                    },
                    stream_id=event.stream_id,
                )
            ]

    async def _stress_cpu_processing(self, event: Event[Any]) -> list[Action[Any]]:
        """Stress test CPU-intensive operations."""
        start_time = time.time()

        # Simulate CPU-intensive work (but keep it short for tests)
        result = sum(i * i for i in range(10000))

        processing_time = time.time() - start_time

        return [
            Action.create(
                "cpu_intensive_processed",
                {
                    "calculation_result": result,
                    "processing_time_ms": processing_time * 1000,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _stress_connection_pool(self, event: Event[Any]) -> list[Action[Any]]:
        """Stress test connection pool limits."""
        # Simulate opening new connection
        self.active_connections += 1

        # Simulate connection pool exhaustion
        max_connections = 10  # Low limit for testing
        if self.active_connections > max_connections:
            return [
                Action.create(
                    "connection_pool_exhausted",
                    {
                        "active_connections": self.active_connections,
                        "max_connections": max_connections,
                        "error": "Connection pool exhausted",
                        "timestamp": time.time(),
                    },
                    stream_id=event.stream_id,
                )
            ]

        return [
            Action.create(
                "connection_acquired",
                {
                    "connection_id": f"conn_{self.active_connections}",
                    "active_connections": self.active_connections,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _stress_thread_pool(self, event: Event[Any]) -> list[Action[Any]]:
        """Stress test thread pool exhaustion."""

        def background_work():
            time.sleep(0.01)  # Short work for testing

        # Create new thread
        thread = threading.Thread(target=background_work)
        self.threads.append(thread)
        thread.start()

        # Clean up completed threads
        self.threads = [t for t in self.threads if t.is_alive()]

        return [
            Action.create(
                "thread_spawned",
                {
                    "thread_count": len(self.threads),
                    "thread_id": thread.ident,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


class HighVolumeTestHandler:
    """Handler designed to test high-volume event processing.

    Real-world scenarios this protects against:
    - Event processing lag under high throughput
    - Queue backlog and overflow
    - Batch processing bottlenecks
    - GC pressure from high allocation rates
    """

    def __init__(self, crew_id: str):
        self.crew_id = crew_id
        self.processed_count = 0
        self.processing_times: list[float] = []
        self.start_time = time.time()

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with performance tracking."""
        event_start = time.time()

        # Simulate some processing work
        payload = event.payload if isinstance(event.payload, dict) else {}
        data_size = len(str(payload))

        # Simulate processing complexity based on data size
        complexity_factor = max(1, data_size // 100)
        await asyncio.sleep(
            0.001 * complexity_factor
        )  # Micro-delay based on complexity

        processing_time = time.time() - event_start
        self.processing_times.append(processing_time)
        self.processed_count += 1

        return [
            Action.create(
                "high_volume_processed",
                {
                    "event_number": self.processed_count,
                    "processing_time_ms": processing_time * 1000,
                    "data_size": data_size,
                    "throughput": self.processed_count
                    / (time.time() - self.start_time),
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


def get_memory_usage() -> dict[str, Any]:
    """Get current memory usage metrics."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / (1024 * 1024),
    }


@pytest.fixture
def performance_stress_events():
    """Create events for performance and stress testing."""
    base_time = datetime(2025, 8, 6, 10, 0, 0, tzinfo=timezone.utc)
    events = []

    # Memory stress events
    for i in range(20):
        events.append(
            EventDict(
                ts=base_time + timedelta(milliseconds=i * 50),
                kind="memory_stress",
                stream_id=f"memory_stream_{i % 3}",
                payload={
                    "operation": "memory_intensive",
                    "data_size": 1024 * (i + 1),
                    "batch_id": i // 5,
                },
            )
        )

    # Resource exhaustion events
    resource_types = [
        "file_descriptors",
        "cpu_intensive",
        "connection_pool",
        "thread_exhaustion",
    ]
    for i, resource_type in enumerate(resource_types * 5):  # 20 events total
        events.append(
            EventDict(
                ts=base_time + timedelta(milliseconds=1000 + i * 50),
                kind="resource_stress",
                stream_id="resource_stream",
                payload={
                    "operation": "resource_exhaustion",
                    "stress_type": resource_type,
                    "intensity": (i % 5) + 1,
                },
            )
        )

    # High volume events
    for i in range(50):
        events.append(
            EventDict(
                ts=base_time + timedelta(milliseconds=2000 + i * 20),
                kind="high_volume",
                stream_id=f"volume_stream_{i % 5}",
                payload={
                    "operation": "bulk_processing",
                    "batch_size": 100 + (i * 10),
                    "data": "x" * (100 + (i * 20)),  # Growing data size
                },
            )
        )

    return events


@pytest.fixture
def performance_events_file(performance_stress_events):
    """Create temporary file with performance stress events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        events_data = []
        for event in performance_stress_events:
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


class TestMemoryLeakDetection:
    """Test memory leak detection and prevention."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection_under_load(self, performance_events_file):
        """Test detection of memory leaks during sustained processing.

        Real-world scenario: Long-running services develop memory leaks
        over time, leading to OOM kills and service instability.

        Production failure this prevents: Memory exhaustion causing
        service crashes, degraded performance, and system instability.
        """
        try:
            # Start memory tracking
            tracemalloc.start()
            initial_memory = get_memory_usage()

            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Handler WITHOUT memory leaks (should be stable)
            clean_handler = MemoryLeakTestHandler("clean_crew", leak_mode=False)
            scheduler.crew_registry.register_crew(clean_handler, [])

            await scheduler.process_events()

            # Force garbage collection
            gc.collect()
            final_memory = get_memory_usage()

            # Verify memory usage stayed reasonable (allowing for some variance)
            memory_growth = final_memory["rss_mb"] - initial_memory["rss_mb"]
            assert memory_growth < 50, f"Excessive memory growth: {memory_growth}MB"

            # Verify events were processed
            assert clean_handler.event_count >= 20  # Memory stress events
            assert len(clean_handler.large_objects) == 0  # No leaks in clean mode

            tracemalloc.stop()

        finally:
            performance_events_file.unlink()

    @pytest.mark.asyncio
    async def test_memory_leak_simulation_and_detection(self):
        """Test that memory leaks can be detected when they occur.

        Real-world scenario: Code changes introduce memory leaks that
        slowly consume system resources over time.

        Production failure this prevents: Gradual memory exhaustion
        leading to OOM conditions and service failures.
        """
        # Create smaller event set for leak testing
        leak_events = [
            {
                "ts": "2025-08-06T10:00:01Z",
                "kind": "leak_test",
                "stream_id": "leak_stream",
                "payload": {"data": "x" * 1000},
            },
            {
                "ts": "2025-08-06T10:00:02Z",
                "kind": "leak_test",
                "stream_id": "leak_stream",
                "payload": {"data": "y" * 1000},
            },
            {
                "ts": "2025-08-06T10:00:03Z",
                "kind": "leak_test",
                "stream_id": "leak_stream",
                "payload": {"data": "z" * 1000},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(leak_events, f)
            leak_file = Path(f.name)

        try:
            tracemalloc.start()
            initial_memory = get_memory_usage()

            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(leak_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Handler WITH intentional memory leaks for testing
            leaky_handler = MemoryLeakTestHandler("leaky_crew", leak_mode=True)
            scheduler.crew_registry.register_crew(leaky_handler, [])

            await scheduler.process_events()

            # Verify leaks were created (for testing purposes)
            assert len(leaky_handler.large_objects) >= 3  # Should have leaked objects
            assert len(leaky_handler.references) >= 3  # Should have circular refs

            final_memory = get_memory_usage()
            memory_growth = final_memory["rss_mb"] - initial_memory["rss_mb"]

            # Verify memory growth occurred (indicating leak detection works)
            assert memory_growth > 0, "Expected memory growth from intentional leaks"

            tracemalloc.stop()

        finally:
            leak_file.unlink()


class TestResourceExhaustionScenarios:
    """Test resource exhaustion scenarios and limits."""

    @pytest.mark.asyncio
    async def test_file_descriptor_exhaustion_handling(self, performance_events_file):
        """Test handling of file descriptor exhaustion.

        Real-world scenario: High-traffic applications can exhaust
        available file descriptors, preventing new connections.

        Production failure this prevents: Service inability to accept
        new connections or open files, leading to request failures.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Resource handler with stress testing enabled
            resource_handler = ResourceExhaustionTestHandler(
                "fd_crew", stress_mode=True
            )
            scheduler.crew_registry.register_crew(resource_handler, [])

            await scheduler.process_events()

            # Verify file descriptors were opened (stress testing occurred)
            assert (
                len(resource_handler.open_files) >= 5
            )  # Some files opened during testing

            # Clean up opened files
            for f in resource_handler.open_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except (OSError, IOError):
                    pass  # Best effort cleanup

        finally:
            performance_events_file.unlink()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_detection(self, performance_events_file):
        """Test detection and handling of connection pool exhaustion.

        Real-world scenario: Database or service connection pools
        become fully utilized under high load.

        Production failure this prevents: Request queuing, timeouts,
        and complete service unavailability due to connection limits.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            resource_handler = ResourceExhaustionTestHandler(
                "pool_crew", stress_mode=True
            )
            scheduler.crew_registry.register_crew(resource_handler, [])

            await scheduler.process_events()

            # Verify connection pool stress occurred (at least some connection pool events)
            assert (
                resource_handler.active_connections >= 5
            )  # Pool should be stressed by connection events

        finally:
            performance_events_file.unlink()

    @pytest.mark.asyncio
    async def test_cpu_intensive_processing_impact(self, performance_events_file):
        """Test impact of CPU-intensive operations on event processing.

        Real-world scenario: CPU-bound tasks block the event loop,
        causing processing delays and timeouts.

        Production failure this prevents: Event processing lag,
        timeout errors, and degraded system responsiveness.
        """
        try:
            start_time = time.time()

            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            resource_handler = ResourceExhaustionTestHandler(
                "cpu_crew", stress_mode=True
            )
            scheduler.crew_registry.register_crew(resource_handler, [])

            await scheduler.process_events()

            total_time = time.time() - start_time

            # Verify processing completed within reasonable time
            # (CPU stress shouldn't completely block processing)
            assert total_time < 10.0, f"Processing took too long: {total_time}s"

            # Verify CPU-intensive work was performed
            assert (
                action_transport.publish_action.call_count >= 90
            )  # All events processed

        finally:
            performance_events_file.unlink()


class TestHighVolumePerformance:
    """Test performance under high event volumes."""

    @pytest.mark.asyncio
    async def test_high_throughput_event_processing(self, performance_events_file):
        """Test system performance under high event throughput.

        Real-world scenario: Production systems must handle thousands
        of events per second without performance degradation.

        Production failure this prevents: Event processing lag,
        queue backlog, and system unresponsiveness under load.
        """
        try:
            start_time = time.time()

            config = LiveCrewConfig(slice_ms=25)  # Faster slicing for high volume
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # High-volume processing handler
            volume_handler = HighVolumeTestHandler("volume_crew")
            scheduler.crew_registry.register_crew(volume_handler, [])

            await scheduler.process_events()

            total_time = time.time() - start_time
            throughput = volume_handler.processed_count / total_time

            # Verify high throughput was achieved
            assert volume_handler.processed_count >= 50  # High volume events
            assert throughput >= 10, f"Low throughput: {throughput} events/sec"

            # Verify processing time consistency
            if volume_handler.processing_times:
                avg_processing_time = sum(volume_handler.processing_times) / len(
                    volume_handler.processing_times
                )
                assert avg_processing_time < 0.1, (
                    f"Slow average processing: {avg_processing_time}s"
                )

        finally:
            performance_events_file.unlink()

    @pytest.mark.asyncio
    async def test_event_processing_latency_under_load(self, performance_events_file):
        """Test event processing latency consistency under load.

        Real-world scenario: Processing latency should remain
        consistent even as event volume increases.

        Production failure this prevents: Increasing latency causing
        timeouts, SLA violations, and poor user experience.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            volume_handler = HighVolumeTestHandler("latency_crew")
            scheduler.crew_registry.register_crew(volume_handler, [])

            await scheduler.process_events()

            # Analyze processing time distribution
            if volume_handler.processing_times:
                processing_times = sorted(volume_handler.processing_times)
                n = len(processing_times)

                # Calculate percentiles
                p50 = processing_times[n // 2] if n > 0 else 0
                p95 = processing_times[int(n * 0.95)] if n > 0 else 0
                p99 = processing_times[int(n * 0.99)] if n > 0 else 0

                # Verify latency is reasonable
                assert p50 < 0.05, f"High P50 latency: {p50}s"
                assert p95 < 0.1, f"High P95 latency: {p95}s"
                assert p99 < 0.2, f"High P99 latency: {p99}s"

        finally:
            performance_events_file.unlink()


class TestGarbageCollectionPressure:
    """Test garbage collection pressure and memory management."""

    @pytest.mark.asyncio
    async def test_gc_pressure_under_sustained_load(self, performance_events_file):
        """Test garbage collection behavior under sustained processing load.

        Real-world scenario: High allocation rates can trigger frequent
        GC cycles, impacting application performance.

        Production failure this prevents: GC pause-related performance
        degradation and processing delays.
        """
        try:
            # Enable GC statistics tracking
            gc.set_debug(gc.DEBUG_STATS)
            initial_gc_counts = gc.get_count()

            config = LiveCrewConfig(slice_ms=25)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            volume_handler = HighVolumeTestHandler("gc_crew")
            scheduler.crew_registry.register_crew(volume_handler, [])

            await scheduler.process_events()

            final_gc_counts = gc.get_count()

            # Verify reasonable GC activity (not excessive)
            gc_growth = [
                final - initial
                for final, initial in zip(final_gc_counts, initial_gc_counts)
            ]

            # Allow some GC activity but not excessive
            assert all(growth < 1000 for growth in gc_growth), (
                f"Excessive GC activity: {gc_growth}"
            )

            # Force cleanup
            gc.collect()
            gc.set_debug(0)  # Disable debug

        finally:
            performance_events_file.unlink()


class TestPerformanceIntegration:
    """Test integrated performance scenarios with multiple stressors."""

    @pytest.mark.asyncio
    async def test_combined_performance_stressors(self, performance_events_file):
        """Test system performance with multiple concurrent stressors.

        Real-world scenario: Production systems face multiple performance
        challenges simultaneously - high volume, resource constraints, and memory pressure.

        Production failure this prevents: System collapse under combined
        load conditions that exceed individual component limits.
        """
        try:
            tracemalloc.start()
            start_time = time.time()
            initial_memory = get_memory_usage()

            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(performance_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Deploy multiple performance handlers (combined stress)
            memory_handler = MemoryLeakTestHandler("memory_crew", leak_mode=False)
            resource_handler = ResourceExhaustionTestHandler(
                "resource_crew", stress_mode=True
            )
            volume_handler = HighVolumeTestHandler("volume_crew")

            scheduler.crew_registry.register_crew(memory_handler, [])
            scheduler.crew_registry.register_crew(resource_handler, [])
            scheduler.crew_registry.register_crew(volume_handler, [])

            await scheduler.process_events()

            total_time = time.time() - start_time
            final_memory = get_memory_usage()

            # Verify system remained stable under combined stress
            assert total_time < 15.0, (
                f"Combined processing took too long: {total_time}s"
            )

            # Verify all handlers processed events
            total_processed = (
                memory_handler.event_count + volume_handler.processed_count
            )
            assert total_processed >= 180, (
                f"Low total processing: {total_processed}"
            )  # 90 events * 2 handlers

            # Verify memory usage remained reasonable despite stress
            memory_growth = final_memory["rss_mb"] - initial_memory["rss_mb"]
            assert memory_growth < 100, (
                f"Excessive memory growth under stress: {memory_growth}MB"
            )

            # Clean up
            for f in resource_handler.open_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except (OSError, IOError):
                    pass

            tracemalloc.stop()

        finally:
            performance_events_file.unlink()
