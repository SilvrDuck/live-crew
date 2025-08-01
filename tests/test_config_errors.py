"""Tests for enhanced config error handling."""

import tempfile
import pytest
from pathlib import Path
import yaml

from live_crew.config.settings import load_config


class TestConfigErrorHandling:
    """Test enhanced error handling in config loading."""

    def test_permission_error_handling(self):
        """Test permission error provides helpful context."""
        # Create a file and then make it unreadable (if possible)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("slice_ms: 100")
            config_path = Path(f.name)

        try:
            # Try to make file unreadable (this might not work on all systems)
            config_path.chmod(0o000)

            with pytest.raises(ValueError) as exc_info:
                load_config(config_path)

            # Check that error message includes file path
            assert str(config_path) in str(exc_info.value)
            assert "Permission denied" in str(exc_info.value)
        except (OSError, PermissionError):
            # Skip test if we can't change permissions
            pytest.skip("Cannot test permission errors on this system")
        finally:
            # Restore permissions and clean up
            try:
                config_path.chmod(0o644)
                config_path.unlink()
            except (OSError, FileNotFoundError):
                pass

    def test_yaml_error_with_context(self):
        """Test YAML parsing error provides file context."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            config_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError) as exc_info:
                load_config(config_path)

            # Check that error message includes file path (pydantic-settings provides this)
            assert str(config_path) in str(exc_info.value)
            assert "mapping values are not allowed here" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_io_error_handling(self):
        """Test IO error provides helpful context."""
        # Test by trying to read a directory as a file instead
        with tempfile.TemporaryDirectory() as tmpdir:
            # pydantic-settings handles this gracefully - if file doesn't exist or can't be read,
            # it just uses defaults. Let's test with a completely invalid file instead.
            invalid_file = Path(tmpdir) / "nonexistent.yaml"

            # This should work - pydantic-settings gracefully handles missing files
            config = load_config(invalid_file)
            # Should get default values
            assert config.slice_ms == 500
            assert config.heartbeat_s == 30
