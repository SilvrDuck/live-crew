"""Tests for event validation configuration and timestamp validation."""

import pytest
from datetime import datetime, timedelta, timezone
from freezegun import freeze_time

from live_crew.config.settings import EventValidationConfig, load_config
from live_crew.core.models import validate_timestamp_field
from tests.utils import EventDict


class TestEventValidationConfig:
    """Test EventValidationConfig model."""

    def test_default_configuration(self):
        """Test default validation configuration."""
        config = EventValidationConfig()

        assert config.timestamp_window_days == 30
        assert config.future_tolerance_seconds == 60
        assert config.strict_mode is True

    def test_custom_configuration(self):
        """Test custom validation configuration."""
        config = EventValidationConfig(
            timestamp_window_days=7, future_tolerance_seconds=120, strict_mode=False
        )

        assert config.timestamp_window_days == 7
        assert config.future_tolerance_seconds == 120
        assert config.strict_mode is False

    def test_timestamp_validation_config(self):
        """Test timestamp validation configuration options."""
        # Test enabled validation (default)
        config = EventValidationConfig()
        assert config.timestamp_validation_enabled is True

        # Test disabled validation
        config = EventValidationConfig(timestamp_validation_enabled=False)
        assert config.timestamp_validation_enabled is False

    def test_validation_constraints(self):
        """Test validation configuration constraints."""
        # timestamp_window_days must be >= 0 (no more -1 support)
        with pytest.raises(ValueError):
            EventValidationConfig(timestamp_window_days=-1)

        # future_tolerance_seconds must be >= 0 and <= 3600
        with pytest.raises(ValueError):
            EventValidationConfig(future_tolerance_seconds=-1)

        with pytest.raises(ValueError):
            EventValidationConfig(future_tolerance_seconds=3601)


class TestTimestampValidation:
    """Test timestamp validation function."""

    @freeze_time("2025-07-20T12:00:00Z")
    def test_valid_timestamp_recent(self):
        """Test validation of recent timestamp."""
        # 5 minutes ago should be valid
        timestamp = datetime(2025, 7, 20, 11, 55, 0, tzinfo=timezone.utc)

        result = validate_timestamp_field(timestamp)
        assert result == timestamp

    @freeze_time("2025-07-20T12:00:00Z")
    def test_valid_timestamp_edge_cases(self):
        """Test validation at edge cases."""
        # Exactly at future tolerance limit (60 seconds)
        future_timestamp = datetime(2025, 7, 20, 12, 1, 0, tzinfo=timezone.utc)
        result = validate_timestamp_field(future_timestamp)
        assert result == future_timestamp

        # Exactly at past window limit (30 days)
        past_timestamp = datetime(2025, 6, 20, 12, 0, 0, tzinfo=timezone.utc)
        result = validate_timestamp_field(past_timestamp)
        assert result == past_timestamp

    @freeze_time("2025-07-20T12:00:00Z")
    def test_invalid_future_timestamp(self):
        """Test validation of future timestamp beyond tolerance."""
        # 2 minutes in the future (beyond 60s tolerance)
        timestamp = datetime(2025, 7, 20, 12, 2, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="cannot be more than 60s in the future"):
            validate_timestamp_field(timestamp)

    @freeze_time("2025-07-20T12:00:00Z")
    def test_invalid_past_timestamp(self):
        """Test validation of timestamp beyond past window."""
        # 31 days ago (beyond 30 day window)
        timestamp = datetime(2025, 6, 19, 12, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="cannot be more than 30 days in the past"):
            validate_timestamp_field(timestamp)

    @freeze_time("2025-07-20T12:00:00Z")
    def test_custom_future_tolerance(self):
        """Test validation with custom future tolerance using env vars."""
        # Test with default config (60s tolerance)
        timestamp = datetime(
            2025, 7, 20, 12, 4, 0, tzinfo=timezone.utc
        )  # 4 minutes future
        with pytest.raises(ValueError, match="cannot be more than 60s in the future"):
            validate_timestamp_field(timestamp)

    @freeze_time("2025-07-20T12:00:00Z")
    def test_custom_past_window(self):
        """Test validation with custom past window using default config."""
        # Test with default config (30 days window)
        timestamp = datetime(2025, 6, 10, 12, 0, 0, tzinfo=timezone.utc)  # 40 days ago
        with pytest.raises(ValueError, match="cannot be more than 30 days in the past"):
            validate_timestamp_field(timestamp)

    @freeze_time("2025-07-20T12:00:00Z")
    def test_disabled_timestamp_validation(self):
        """Test disabled timestamp validation."""
        from live_crew.config.settings import LiveCrewConfig

        config = LiveCrewConfig(
            event_validation=EventValidationConfig(timestamp_validation_enabled=False)
        )

        # Very old timestamp should be valid when validation is disabled
        past_timestamp = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = validate_timestamp_field(past_timestamp, config)
        assert result == past_timestamp

        # Future timestamp should also be valid when validation is disabled
        future_timestamp = datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = validate_timestamp_field(future_timestamp, config)
        assert result == future_timestamp

    def test_timezone_conversion(self):
        """Test timezone conversion in validation."""
        # Naive datetime should be converted to UTC
        naive_timestamp = datetime(2025, 7, 20, 12, 0, 0)
        result = validate_timestamp_field(naive_timestamp)
        assert result.tzinfo == timezone.utc

        # Non-UTC timezone should be converted to UTC
        from datetime import timezone as tz

        est = tz(timedelta(hours=-5))
        est_timestamp = datetime(
            2025, 7, 20, 7, 0, 0, tzinfo=est
        )  # 7 AM EST = 12 PM UTC
        result = validate_timestamp_field(est_timestamp)
        assert result.tzinfo == timezone.utc
        assert result.hour == 12  # Converted to UTC

    def test_default_config_fallback(self):
        """Test that default config is used when none provided."""
        # Should not raise an error and use default config
        timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = validate_timestamp_field(timestamp)  # No config provided
        assert result.tzinfo == timezone.utc


class TestEventModelValidation:
    """Test Event model with validation configuration."""

    @freeze_time("2025-07-20T12:00:00Z")
    def test_event_with_valid_timestamp(self):
        """Test Event creation with valid timestamp."""
        event_data = {
            "ts": "2025-07-20T11:55:00Z",  # 5 minutes ago
            "kind": "test_event",
            "stream_id": "test_stream",
            "payload": {"test": "data"},
        }

        event = EventDict(**event_data)
        assert event.kind == "test_event"
        assert event.ts.tzinfo == timezone.utc

    @freeze_time("2025-07-20T12:00:00Z")
    def test_event_with_invalid_timestamp(self):
        """Test Event creation with invalid timestamp."""
        event_data = {
            "ts": "2025-06-01T12:00:00Z",  # More than 30 days ago
            "kind": "test_event",
            "stream_id": "test_stream",
            "payload": {"test": "data"},
        }

        with pytest.raises(ValueError, match="cannot be more than 30 days in the past"):
            EventDict(**event_data)


class TestConfigurationIntegration:
    """Test integration of event validation config with main config."""

    def test_default_config_includes_event_validation(self):
        """Test that default config includes event validation."""
        config = load_config()

        assert hasattr(config, "event_validation")
        assert config.event_validation.timestamp_window_days == 30
        assert config.event_validation.future_tolerance_seconds == 60
        assert config.event_validation.strict_mode is True

    def test_custom_event_validation_config(self):
        """Test custom event validation in YAML config."""
        import tempfile
        import yaml
        from pathlib import Path

        # Create temporary config file
        config_data = {
            "slice_ms": 1000,
            "heartbeat_s": 60,
            "kv_backend": "memory",
            "event_validation": {
                "timestamp_window_days": 7,
                "future_tolerance_seconds": 120,
                "strict_mode": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)

            assert config.event_validation.timestamp_window_days == 7
            assert config.event_validation.future_tolerance_seconds == 120
            assert config.event_validation.strict_mode is False
        finally:
            temp_path.unlink()  # Cleanup
