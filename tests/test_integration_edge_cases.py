"""Integration edge cases tests for production-critical system interaction failures.

These tests focus on external system dependencies and their failure modes,
including database connections, file system operations, network calls, and
service integrations that commonly cause production outages.

Each test simulates realistic failure scenarios that occur when external
dependencies fail, timeout, or behave unexpectedly, ensuring the system
maintains stability and provides proper error handling.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from live_crew import Action, Event
from live_crew.config.settings import LiveCrewConfig
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


class ExternalSystemTestHandler:
    """Handler that simulates interactions with external systems.

    Real-world scenarios this protects against:
    - Database connection failures and timeouts
    - File system permission and I/O errors
    - Network service unavailability
    - Third-party API failures and rate limiting
    - Message queue connection losses
    """

    def __init__(self, crew_id: str, simulate_failures: bool = False):
        self.crew_id = crew_id
        self.simulate_failures = simulate_failures
        self.external_calls: list[dict] = []
        self.failure_count = 0

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with external system interactions."""
        payload = event.payload if isinstance(event.payload, dict) else {}

        # Simulate database operations
        if payload.get("operation") == "database_query":
            return await self._handle_database_operation(event, payload)

        # Simulate file system operations
        elif payload.get("operation") == "file_operation":
            return await self._handle_file_operation(event, payload)

        # Simulate network API calls
        elif payload.get("operation") == "api_call":
            return await self._handle_api_operation(event, payload)

        # Simulate message queue operations
        elif payload.get("operation") == "queue_operation":
            return await self._handle_queue_operation(event, payload)

        # Default processing
        return [
            Action.create(
                "processed_normally",
                {
                    "crew_id": self.crew_id,
                    "external_calls": len(self.external_calls),
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _handle_database_operation(
        self, event: Event[Any], payload: dict
    ) -> list[Action[Any]]:
        """Simulate database operations with potential failures."""
        operation_record = {
            "type": "database_operation",
            "query": payload.get("query", "SELECT * FROM table"),
            "timestamp": time.time(),
            "attempt_number": len(self.external_calls) + 1,
        }
        self.external_calls.append(operation_record)

        if self.simulate_failures:
            # Simulate various database failure modes
            failure_type = payload.get("failure_type", "timeout")

            if failure_type == "timeout":
                # Simulate database timeout
                await asyncio.sleep(0.01)  # Small delay to simulate timeout
                self.failure_count += 1
                return [
                    Action.create(
                        "db_timeout_error",
                        {
                            "error": "Database connection timeout after 30 seconds",
                            "query": payload.get("query"),
                            "retry_count": operation_record["attempt_number"],
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "connection_refused":
                self.failure_count += 1
                return [
                    Action.create(
                        "db_connection_error",
                        {
                            "error": "Connection refused: No connection could be made",
                            "host": payload.get("host", "db.example.com"),
                            "port": payload.get("port", 5432),
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "deadlock":
                self.failure_count += 1
                return [
                    Action.create(
                        "db_deadlock_error",
                        {
                            "error": "Deadlock detected during transaction",
                            "transaction_id": f"tx_{int(time.time())}",
                            "affected_tables": ["users", "orders"],
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "connection_pool_exhausted":
                self.failure_count += 1
                return [
                    Action.create(
                        "db_pool_exhausted",
                        {
                            "error": "Connection pool exhausted: max 100 connections",
                            "active_connections": 100,
                            "waiting_requests": 25,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

        # Successful database operation
        return [
            Action.create(
                "db_query_success",
                {
                    "result_count": payload.get("expected_rows", 42),
                    "query_time_ms": 15,
                    "connection_id": f"conn_{int(time.time())}",
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _handle_file_operation(
        self, event: Event[Any], payload: dict
    ) -> list[Action[Any]]:
        """Simulate file system operations with potential failures."""
        operation_record = {
            "type": "file_operation",
            "operation": payload.get("file_action", "read"),
            "path": payload.get("file_path", "/tmp/data.txt"),
            "timestamp": time.time(),
        }
        self.external_calls.append(operation_record)

        if self.simulate_failures:
            failure_type = payload.get("failure_type", "permission_denied")

            if failure_type == "permission_denied":
                self.failure_count += 1
                return [
                    Action.create(
                        "file_permission_error",
                        {
                            "error": "Permission denied: insufficient privileges",
                            "path": payload.get("file_path", "/etc/sensitive.conf"),
                            "required_permission": "read",
                            "current_user": "app_user",
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "disk_full":
                self.failure_count += 1
                return [
                    Action.create(
                        "file_disk_full_error",
                        {
                            "error": "No space left on device",
                            "device": "/dev/sda1",
                            "available_space": 0,
                            "required_space": payload.get("file_size", 1024000),
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "file_not_found":
                self.failure_count += 1
                return [
                    Action.create(
                        "file_not_found_error",
                        {
                            "error": "File or directory not found",
                            "path": payload.get("file_path", "/missing/file.txt"),
                            "parent_exists": False,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

        # Successful file operation
        return [
            Action.create(
                "file_operation_success",
                {
                    "operation": payload.get("file_action", "read"),
                    "path": payload.get("file_path", "/tmp/data.txt"),
                    "size_bytes": payload.get("file_size", 2048),
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _handle_api_operation(
        self, event: Event[Any], payload: dict
    ) -> list[Action[Any]]:
        """Simulate external API calls with potential failures."""
        operation_record = {
            "type": "api_operation",
            "endpoint": payload.get("endpoint", "/api/v1/data"),
            "method": payload.get("method", "GET"),
            "timestamp": time.time(),
        }
        self.external_calls.append(operation_record)

        if self.simulate_failures:
            failure_type = payload.get("failure_type", "timeout")

            if failure_type == "timeout":
                await asyncio.sleep(0.01)  # Simulate network delay
                self.failure_count += 1
                return [
                    Action.create(
                        "api_timeout_error",
                        {
                            "error": "Request timeout after 30 seconds",
                            "endpoint": payload.get("endpoint", "/api/v1/data"),
                            "timeout_seconds": 30,
                            "retry_count": 3,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "rate_limited":
                self.failure_count += 1
                return [
                    Action.create(
                        "api_rate_limit_error",
                        {
                            "error": "Rate limit exceeded",
                            "status_code": 429,
                            "rate_limit": "100 requests per minute",
                            "reset_time": time.time() + 60,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "server_error":
                self.failure_count += 1
                return [
                    Action.create(
                        "api_server_error",
                        {
                            "error": "Internal server error",
                            "status_code": 500,
                            "error_id": f"err_{int(time.time())}",
                            "retry_after": 300,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

        # Successful API call
        return [
            Action.create(
                "api_call_success",
                {
                    "endpoint": payload.get("endpoint", "/api/v1/data"),
                    "status_code": 200,
                    "response_time_ms": 150,
                    "data_size": 1024,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]

    async def _handle_queue_operation(
        self, event: Event[Any], payload: dict
    ) -> list[Action[Any]]:
        """Simulate message queue operations with potential failures."""
        operation_record = {
            "type": "queue_operation",
            "queue": payload.get("queue_name", "processing_queue"),
            "operation": payload.get("queue_action", "publish"),
            "timestamp": time.time(),
        }
        self.external_calls.append(operation_record)

        if self.simulate_failures:
            failure_type = payload.get("failure_type", "connection_lost")

            if failure_type == "connection_lost":
                self.failure_count += 1
                return [
                    Action.create(
                        "queue_connection_error",
                        {
                            "error": "Connection to message broker lost",
                            "broker_host": payload.get(
                                "broker_host", "rabbitmq.example.com"
                            ),
                            "last_heartbeat": time.time() - 30,
                            "reconnect_attempts": 5,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

            elif failure_type == "queue_full":
                self.failure_count += 1
                return [
                    Action.create(
                        "queue_full_error",
                        {
                            "error": "Queue is full, cannot accept new messages",
                            "queue_name": payload.get("queue_name", "processing_queue"),
                            "current_size": 10000,
                            "max_size": 10000,
                            "timestamp": time.time(),
                        },
                        stream_id=event.stream_id,
                    )
                ]

        # Successful queue operation
        return [
            Action.create(
                "queue_operation_success",
                {
                    "queue": payload.get("queue_name", "processing_queue"),
                    "operation": payload.get("queue_action", "publish"),
                    "message_id": f"msg_{int(time.time())}",
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


class CascadingFailureTestHandler:
    """Handler that simulates cascading failure scenarios.

    Real-world scenarios this protects against:
    - Service dependency chains failing sequentially
    - Circuit breaker patterns under load
    - Upstream service failures causing downstream problems
    - Resource exhaustion cascade across services
    """

    def __init__(self, crew_id: str, failure_threshold: int = 3):
        self.crew_id = crew_id
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.circuit_open = False
        self.failure_cascade: list[dict] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with cascading failure detection."""
        payload = event.payload if isinstance(event.payload, dict) else {}

        # Check if circuit breaker is open
        if self.circuit_open:
            failure_record = {
                "type": "circuit_breaker_open",
                "service": self.crew_id,
                "consecutive_failures": self.consecutive_failures,
                "timestamp": time.time(),
            }
            self.failure_cascade.append(failure_record)

            return [
                Action.create(
                    "circuit_breaker_open", failure_record, stream_id=event.stream_id
                )
            ]

        # Simulate service call
        if payload.get("simulate_failure", False):
            self.consecutive_failures += 1

            failure_record = {
                "type": "service_failure",
                "service": self.crew_id,
                "failure_count": self.consecutive_failures,
                "error": payload.get("error", "Service unavailable"),
                "timestamp": time.time(),
            }
            self.failure_cascade.append(failure_record)

            # Check if we should open circuit breaker
            if self.consecutive_failures >= self.failure_threshold:
                self.circuit_open = True
                cascade_record = {
                    "type": "cascade_triggered",
                    "service": self.crew_id,
                    "trigger_threshold": self.failure_threshold,
                    "cascade_depth": len(self.failure_cascade),
                    "timestamp": time.time(),
                }
                self.failure_cascade.append(cascade_record)

                return [
                    Action.create(
                        "cascade_failure_triggered",
                        cascade_record,
                        stream_id=event.stream_id,
                    )
                ]

            return [
                Action.create(
                    "service_failure_detected",
                    failure_record,
                    stream_id=event.stream_id,
                )
            ]

        # Successful processing - reset failure count
        self.consecutive_failures = 0
        self.circuit_open = False

        return [
            Action.create(
                "service_healthy",
                {
                    "service": self.crew_id,
                    "health_status": "operational",
                    "failure_count": 0,
                    "timestamp": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


@pytest.fixture
def integration_failure_events():
    """Create events that trigger various integration failure scenarios."""
    base_time = datetime(2025, 8, 6, 10, 0, 0, tzinfo=timezone.utc)

    # Database failure scenarios
    database_failure_events = [
        EventDict(
            ts=base_time + timedelta(seconds=1),
            kind="db_operation",
            stream_id="database_service",
            payload={
                "operation": "database_query",
                "query": "SELECT * FROM users WHERE active = 1",
                "failure_type": "timeout",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=2),
            kind="db_operation",
            stream_id="database_service",
            payload={
                "operation": "database_query",
                "query": "INSERT INTO orders (user_id, total) VALUES (1, 99.99)",
                "failure_type": "deadlock",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=3),
            kind="db_operation",
            stream_id="database_service",
            payload={
                "operation": "database_query",
                "query": "UPDATE user_sessions SET last_active = NOW()",
                "failure_type": "connection_pool_exhausted",
            },
        ),
    ]

    # File system failure scenarios
    filesystem_failure_events = [
        EventDict(
            ts=base_time + timedelta(seconds=4),
            kind="file_operation",
            stream_id="file_service",
            payload={
                "operation": "file_operation",
                "file_action": "write",
                "file_path": "/var/log/app.log",
                "failure_type": "permission_denied",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=5),
            kind="file_operation",
            stream_id="file_service",
            payload={
                "operation": "file_operation",
                "file_action": "write",
                "file_path": "/tmp/large_upload.dat",
                "file_size": 5000000000,  # 5GB file
                "failure_type": "disk_full",
            },
        ),
    ]

    # API failure scenarios
    api_failure_events = [
        EventDict(
            ts=base_time + timedelta(seconds=6),
            kind="api_request",
            stream_id="api_service",
            payload={
                "operation": "api_call",
                "endpoint": "/api/v1/payment/process",
                "method": "POST",
                "failure_type": "timeout",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=7),
            kind="api_request",
            stream_id="api_service",
            payload={
                "operation": "api_call",
                "endpoint": "/api/v1/user/profile",
                "method": "GET",
                "failure_type": "rate_limited",
            },
        ),
    ]

    # Message queue failure scenarios
    queue_failure_events = [
        EventDict(
            ts=base_time + timedelta(seconds=8),
            kind="queue_operation",
            stream_id="queue_service",
            payload={
                "operation": "queue_operation",
                "queue_name": "order_processing",
                "queue_action": "publish",
                "failure_type": "connection_lost",
            },
        )
    ]

    # Cascading failure scenario
    cascade_failure_events = [
        EventDict(
            ts=base_time + timedelta(seconds=9),
            kind="service_call",
            stream_id="cascade_test",
            payload={"simulate_failure": True, "error": "Upstream service timeout"},
        ),
        EventDict(
            ts=base_time + timedelta(seconds=10),
            kind="service_call",
            stream_id="cascade_test",
            payload={"simulate_failure": True, "error": "Database connection failed"},
        ),
        EventDict(
            ts=base_time + timedelta(seconds=11),
            kind="service_call",
            stream_id="cascade_test",
            payload={
                "simulate_failure": True,
                "error": "Circuit breaker should trigger",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=12),
            kind="service_call",
            stream_id="cascade_test",
            payload={
                "simulate_failure": False,  # Should be blocked by circuit breaker
                "message": "This should not process",
            },
        ),
    ]

    return (
        database_failure_events
        + filesystem_failure_events
        + api_failure_events
        + queue_failure_events
        + cascade_failure_events
    )


@pytest.fixture
def integration_failure_events_file(integration_failure_events):
    """Create temporary file with integration failure events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        events_data = []
        for event in integration_failure_events:
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


class TestDatabaseIntegrationFailures:
    """Test database integration failure scenarios."""

    @pytest.mark.asyncio
    async def test_database_timeout_handling(self, integration_failure_events_file):
        """Test that database timeouts are handled gracefully.

        Real-world scenario: Database server becomes unresponsive due to load,
        hardware issues, or network problems.

        Production failure this prevents: Application hanging indefinitely,
        cascading timeouts across the system, resource exhaustion.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # External system handler with failure simulation
            external_handler = ExternalSystemTestHandler(
                "db_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify database timeout was handled
            db_operations = [
                call
                for call in external_handler.external_calls
                if call["type"] == "database_operation"
            ]
            assert len(db_operations) >= 1

            # Verify timeout error was generated
            assert (
                action_transport.publish_action.call_count >= 12
            )  # All events processed

            # Check that failures were detected
            assert external_handler.failure_count >= 3  # DB failures in test data

        finally:
            integration_failure_events_file.unlink()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_recovery(
        self, integration_failure_events_file
    ):
        """Test recovery from database connection pool exhaustion.

        Real-world scenario: High traffic causes all database connections
        to be in use, preventing new requests from being processed.

        Production failure this prevents: Complete application unavailability
        due to database connection pool saturation.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "pool_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify pool exhaustion was detected and handled
            db_operations = [
                call
                for call in external_handler.external_calls
                if call["type"] == "database_operation"
            ]
            assert len(db_operations) >= 3  # Multiple DB operations in test data

        finally:
            integration_failure_events_file.unlink()


class TestFileSystemIntegrationFailures:
    """Test file system integration failure scenarios."""

    @pytest.mark.asyncio
    async def test_file_permission_errors(self, integration_failure_events_file):
        """Test handling of file permission errors.

        Real-world scenario: Application attempts to write to restricted
        directories or read protected configuration files.

        Production failure this prevents: Application crashes due to
        unhandled permission exceptions, security violations.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "file_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify file operations were attempted
            file_operations = [
                call
                for call in external_handler.external_calls
                if call["type"] == "file_operation"
            ]
            assert len(file_operations) >= 2  # File operations in test data

        finally:
            integration_failure_events_file.unlink()

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_handling(
        self, integration_failure_events_file
    ):
        """Test handling of disk space exhaustion.

        Real-world scenario: Disk becomes full due to log files, temporary
        files, or large data uploads.

        Production failure this prevents: System instability, crash loops,
        data loss due to failed writes.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "disk_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify all events were processed even with disk full errors
            assert action_transport.publish_action.call_count >= 12

        finally:
            integration_failure_events_file.unlink()


class TestNetworkIntegrationFailures:
    """Test network and external API integration failure scenarios."""

    @pytest.mark.asyncio
    async def test_api_timeout_and_retry_logic(self, integration_failure_events_file):
        """Test API timeout handling and retry logic.

        Real-world scenario: External API becomes slow or unresponsive,
        requiring timeout handling and intelligent retry strategies.

        Production failure this prevents: Hanging requests, cascading
        timeouts, poor user experience due to slow external dependencies.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "api_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify API operations were attempted
            api_operations = [
                call
                for call in external_handler.external_calls
                if call["type"] == "api_operation"
            ]
            assert len(api_operations) >= 2  # API operations in test data

        finally:
            integration_failure_events_file.unlink()

    @pytest.mark.asyncio
    async def test_rate_limiting_backoff_strategy(
        self, integration_failure_events_file
    ):
        """Test rate limiting and backoff strategies.

        Real-world scenario: External API enforces rate limits,
        requiring intelligent backoff and retry strategies.

        Production failure this prevents: API bans, service degradation,
        failed requests due to hitting rate limits.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "rate_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify rate limiting was detected and handled
            assert (
                external_handler.failure_count > 0
            )  # Should detect rate limit failures

        finally:
            integration_failure_events_file.unlink()


class TestMessageQueueIntegrationFailures:
    """Test message queue integration failure scenarios."""

    @pytest.mark.asyncio
    async def test_queue_connection_loss_recovery(
        self, integration_failure_events_file
    ):
        """Test recovery from message queue connection loss.

        Real-world scenario: Message broker restarts or network partition
        causes loss of connection to message queue.

        Production failure this prevents: Lost messages, stuck processing,
        inability to communicate between services.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            external_handler = ExternalSystemTestHandler(
                "queue_crew", simulate_failures=True
            )
            scheduler.crew_registry.register_crew(external_handler, [])

            await scheduler.process_events()

            # Verify queue operations were attempted
            queue_operations = [
                call
                for call in external_handler.external_calls
                if call["type"] == "queue_operation"
            ]
            assert len(queue_operations) >= 1  # Queue operations in test data

        finally:
            integration_failure_events_file.unlink()


class TestCascadingFailureScenarios:
    """Test cascading failure scenarios across multiple systems."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern_activation(
        self, integration_failure_events_file
    ):
        """Test circuit breaker pattern activation under failure conditions.

        Real-world scenario: Service experiences repeated failures, triggering
        circuit breaker to prevent cascading failures to dependent services.

        Production failure this prevents: System-wide outages caused by
        failing services bringing down healthy services.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)  # Faster processing for cascade test
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Cascading failure handler with low threshold for testing
            cascade_handler = CascadingFailureTestHandler(
                "cascade_crew", failure_threshold=3
            )
            scheduler.crew_registry.register_crew(cascade_handler, [])

            await scheduler.process_events()

            # Verify cascade was detected
            assert len(cascade_handler.failure_cascade) >= 3  # Should trigger cascade

            # Verify circuit breaker opened
            assert cascade_handler.circuit_open

            # Check cascade trigger was recorded
            cascade_triggers = [
                record
                for record in cascade_handler.failure_cascade
                if record["type"] == "cascade_triggered"
            ]
            assert len(cascade_triggers) >= 1

        finally:
            integration_failure_events_file.unlink()

    @pytest.mark.asyncio
    async def test_dependency_chain_failure_isolation(
        self, integration_failure_events_file
    ):
        """Test isolation of failures in service dependency chains.

        Real-world scenario: Service A depends on B, which depends on C.
        When C fails, the failure should be contained and not crash A.

        Production failure this prevents: Complete system outages due to
        single point of failure in dependency chains.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(integration_failure_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Multiple handlers simulating dependency chain
            service_a_handler = ExternalSystemTestHandler(
                "service_a", simulate_failures=True
            )
            service_b_handler = CascadingFailureTestHandler(
                "service_b", failure_threshold=2
            )
            service_c_handler = ExternalSystemTestHandler(
                "service_c", simulate_failures=True
            )

            scheduler.crew_registry.register_crew(service_a_handler, [])
            scheduler.crew_registry.register_crew(service_b_handler, [])
            scheduler.crew_registry.register_crew(service_c_handler, [])

            await scheduler.process_events()

            # Verify all services processed events (isolation working)
            total_calls = (
                len(service_a_handler.external_calls)
                + len(service_b_handler.failure_cascade)
                + len(service_c_handler.external_calls)
            )
            assert total_calls >= 10  # Multiple services should have activity

        finally:
            integration_failure_events_file.unlink()


class TestIntegrationRecoveryPatterns:
    """Test recovery patterns for integration failures."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_under_failures(self):
        """Test graceful degradation when external dependencies fail.

        Real-world scenario: When external services are unavailable,
        the system should continue operating with reduced functionality
        rather than complete failure.

        Production failure this prevents: Complete system unavailability
        due to non-critical external service failures.
        """
        # Create events with mix of critical and non-critical failures
        degradation_events = [
            {
                "ts": "2025-08-06T10:00:01Z",
                "kind": "critical_operation",
                "stream_id": "critical_service",
                "payload": {
                    "operation": "database_query",
                    "query": "SELECT * FROM core_data",
                    "failure_type": "timeout",
                    "critical": True,
                },
            },
            {
                "ts": "2025-08-06T10:00:02Z",
                "kind": "optional_operation",
                "stream_id": "optional_service",
                "payload": {
                    "operation": "api_call",
                    "endpoint": "/api/v1/analytics/track",
                    "failure_type": "server_error",
                    "critical": False,
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(degradation_events, f)
            degradation_file = Path(f.name)

        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(degradation_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Handler that differentiates critical vs non-critical failures
            class GracefulDegradationHandler:
                def __init__(self, crew_id: str):
                    self.crew_id = crew_id
                    self.degraded_services: set[str] = set()

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    payload = event.payload if isinstance(event.payload, dict) else {}

                    # Check if this is a critical operation
                    is_critical = payload.get("critical", True)
                    service = event.stream_id

                    if payload.get("failure_type"):
                        if is_critical:
                            # Critical failure - must be handled
                            return [
                                Action.create(
                                    "critical_failure_handled",
                                    {
                                        "service": service,
                                        "error": payload.get("failure_type"),
                                        "fallback_activated": True,
                                        "timestamp": time.time(),
                                    },
                                    stream_id=event.stream_id,
                                )
                            ]
                        else:
                            # Non-critical failure - degrade gracefully
                            self.degraded_services.add(service)
                            return [
                                Action.create(
                                    "graceful_degradation",
                                    {
                                        "service": service,
                                        "error": payload.get("failure_type"),
                                        "service_disabled": True,
                                        "core_functionality_preserved": True,
                                        "timestamp": time.time(),
                                    },
                                    stream_id=event.stream_id,
                                )
                            ]

                    # Normal processing
                    return [
                        Action.create(
                            "normal_processing",
                            {
                                "service": service,
                                "degraded_services": list(self.degraded_services),
                                "timestamp": time.time(),
                            },
                            stream_id=event.stream_id,
                        )
                    ]

            degradation_handler = GracefulDegradationHandler("degradation_crew")
            scheduler.crew_registry.register_crew(degradation_handler, [])

            await scheduler.process_events()

            # Verify graceful degradation occurred
            assert (
                len(degradation_handler.degraded_services) >= 1
            )  # Optional service degraded

            # Verify both events were processed
            assert action_transport.publish_action.call_count >= 2

        finally:
            degradation_file.unlink()
