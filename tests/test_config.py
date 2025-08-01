"""Tests for configuration loading and management.

Based on specification in .vibes/live_crew_spec.md section 3.3
"""

import tempfile
import yaml
from pathlib import Path

from live_crew.config.settings import LiveCrewConfig, load_config


class TestLiveCrewConfig:
    """Test cases for LiveCrewConfig model."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = LiveCrewConfig()

        assert config.slice_ms == 500
        assert config.heartbeat_s == 30  # Default heartbeat interval
        assert config.kv_backend == "jetstream"
        assert config.vector is None  # Optional vector config

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = LiveCrewConfig(slice_ms=250, heartbeat_s=3, kv_backend="redis")

        assert config.slice_ms == 250
        assert config.heartbeat_s == 3
        assert config.kv_backend == "redis"

    def test_config_validation_slice_ms(self):
        """Test slice_ms validation."""
        # Valid values
        config = LiveCrewConfig(slice_ms=100)
        assert config.slice_ms == 100

        config = LiveCrewConfig(slice_ms=1000)
        assert config.slice_ms == 1000

        # Invalid values should raise validation error
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LiveCrewConfig(slice_ms=0)  # Too small

        with pytest.raises(ValidationError):
            LiveCrewConfig(slice_ms=10001)  # Too large

    def test_config_validation_heartbeat_s(self):
        """Test heartbeat_s validation."""
        # Valid values
        config = LiveCrewConfig(heartbeat_s=1)
        assert config.heartbeat_s == 1

        config = LiveCrewConfig(heartbeat_s=300)
        assert config.heartbeat_s == 300

        # Invalid values should raise validation error
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LiveCrewConfig(heartbeat_s=0)  # Too small

        with pytest.raises(ValidationError):
            LiveCrewConfig(heartbeat_s=301)  # Too large

    def test_config_validation_kv_backend(self):
        """Test kv_backend validation."""
        # Valid backends
        for backend in ["jetstream", "redis", "memory"]:
            config = LiveCrewConfig(kv_backend=backend)
            assert config.kv_backend == backend

        # Invalid backend should raise validation error
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LiveCrewConfig(kv_backend="invalid_backend")

    def test_config_with_vector_settings(self):
        """Test configuration with vector store settings."""
        vector_config = {"backend": "qdrant", "url": "http://qdrant:6333"}

        config = LiveCrewConfig(vector=vector_config)
        assert config.vector == vector_config
        assert config.vector["backend"] == "qdrant"
        assert config.vector["url"] == "http://qdrant:6333"

    def test_config_immutable(self):
        """Test that config is immutable (frozen)."""
        config = LiveCrewConfig()

        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            config.slice_ms = 1000


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_config_no_file(self):
        """Test loading config when no file exists - should use defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent-config.yaml"
            config = load_config(config_path)

            # Should get default values
            assert config.slice_ms == 500
            assert config.heartbeat_s == 30
            assert config.kv_backend == "jetstream"

    def test_load_config_from_yaml_file(self):
        """Test loading config from YAML file."""
        config_data = {
            "slice_ms": 250,
            "heartbeat_s": 3,
            "kv_backend": "redis",
            "vector": {"backend": "qdrant", "url": "http://qdrant:6333"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert config.slice_ms == 250
            assert config.heartbeat_s == 3
            assert config.kv_backend == "redis"
            assert config.vector == {"backend": "qdrant", "url": "http://qdrant:6333"}
        finally:
            config_path.unlink()

    def test_load_config_partial_yaml(self):
        """Test loading config with partial YAML (missing keys use defaults)."""
        config_data = {
            "slice_ms": 100
            # Other fields missing - should use defaults
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = load_config(config_path)

            assert config.slice_ms == 100  # From file
            assert config.heartbeat_s == 30  # Default
            assert config.kv_backend == "jetstream"  # Default
            assert config.vector is None  # Default
        finally:
            config_path.unlink()

    def test_load_config_invalid_yaml(self):
        """Test loading config with invalid YAML raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            config_path = Path(f.name)

        try:
            import pytest

            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_config_validation_error(self):
        """Test loading config with validation errors."""
        config_data = {
            "slice_ms": -100,  # Invalid value
            "kv_backend": "invalid",  # Invalid backend
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            import pytest
            from pydantic import ValidationError

            with pytest.raises(ValidationError):
                load_config(config_path)
        finally:
            config_path.unlink()

    def test_load_config_default_path(self):
        """Test loading config with default path."""
        # Should not crash when called without arguments
        config = load_config()

        # Should get default values when no config file exists
        assert config.slice_ms == 500
        assert config.heartbeat_s == 30
        assert config.kv_backend == "jetstream"


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_integrates_with_timeslice(self):
        """Test that configuration integrates with slice_index function."""
        from live_crew.core.timeslice import slice_index
        from datetime import datetime, timezone, timedelta

        # Test that config.slice_ms can be used with slice_index
        config = LiveCrewConfig(slice_ms=250)
        assert config.slice_ms == 250

        # Test slice_index uses custom slice_ms
        epoch0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts_250ms = epoch0 + timedelta(milliseconds=250)

        # With 250ms slices, 250ms should be slice 1
        assert slice_index(ts_250ms, epoch0, slice_ms=config.slice_ms) == 1

        # With default 500ms slices, 250ms should be slice 0
        assert slice_index(ts_250ms, epoch0) == 0

    def test_example_config_from_spec(self):
        """Test the exact example configuration from the specification."""
        config_data = {
            "slice_ms": 250,
            "heartbeat_s": 3,
            "kv_backend": "redis",
            "vector": {"backend": "qdrant", "url": "http://qdrant:6333"},
        }

        config = LiveCrewConfig(**config_data)

        assert config.slice_ms == 250
        assert config.heartbeat_s == 3
        assert config.kv_backend == "redis"
        assert config.vector["backend"] == "qdrant"
        assert config.vector["url"] == "http://qdrant:6333"
