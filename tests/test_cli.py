"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path

import pytest
from freezegun import freeze_time
from typer.testing import CliRunner

from live_crew.cli import app


@pytest.fixture
def cli_runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_events_file():
    """Create temporary events file with valid timestamps."""
    events = [
        {
            "ts": "2025-01-15T12:00:00Z",
            "kind": "user_action",
            "stream_id": "user_stream",
            "payload": {
                "user_id": "user123",
                "action": "login",
                "metadata": {"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"},
            },
        },
        {
            "ts": "2025-01-15T12:00:01Z",
            "kind": "data_update",
            "stream_id": "data_stream",
            "payload": {
                "entity_id": "entity456",
                "updates": {"status": "active", "last_seen": "2025-01-15T12:00:01Z"},
            },
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(events, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def temp_config_file():
    """Create temporary config file."""
    config = {"slice_ms": 1000, "heartbeat_s": 60, "kv_backend": "jetstream"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml

        yaml.dump(config, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


class TestCLICommands:
    """Test CLI commands."""

    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "live-crew 0.1.0" in result.stdout
        assert "Low-latency, slice-based orchestration" in result.stdout

    def test_config_show_command(self, cli_runner):
        """Test config show command."""
        result = cli_runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "📋 Current Configuration:" in result.stdout
        assert "Slice duration:" in result.stdout
        assert "Heartbeat:" in result.stdout
        assert "Backend:" in result.stdout

    def test_config_validate_command(self, cli_runner, temp_config_file):
        """Test config validate command."""
        result = cli_runner.invoke(app, ["config", "--validate", str(temp_config_file)])
        assert result.exit_code == 0
        assert f"✅ Configuration file '{temp_config_file}' is valid" in result.stdout

    def test_config_validate_nonexistent_file(self, cli_runner):
        """Test config validate with nonexistent file."""
        result = cli_runner.invoke(app, ["config", "--validate", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_help_command(self, cli_runner):
        """Test help command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Low-latency, slice-based orchestration" in result.stdout
        assert "process" in result.stdout
        assert "config" in result.stdout
        assert "version" in result.stdout


class TestProcessCommand:
    """Test the process command with time-dependent validation."""

    @freeze_time("2025-01-15T12:00:00Z")
    def test_process_command_basic(self, cli_runner, temp_events_file):
        """Test basic process command with frozen time."""
        result = cli_runner.invoke(app, ["process", str(temp_events_file)])
        assert result.exit_code == 0

    @freeze_time("2025-01-15T12:00:00Z")
    def test_process_command_verbose(self, cli_runner, temp_events_file):
        """Test process command with verbose output."""
        result = cli_runner.invoke(app, ["process", str(temp_events_file), "--verbose"])
        assert result.exit_code == 0
        assert "📋 Configuration loaded:" in result.stdout
        assert "Slice duration: 500ms" in result.stdout
        assert "📂 Processing:" in result.stdout
        assert "🔧 Scheduler initialized" in result.stdout
        assert "🎯 Registered crews: 1" in result.stdout
        assert "📊 Processed" in result.stdout
        assert "✅ Processing completed successfully" in result.stdout

    @freeze_time("2025-01-15T12:00:00Z")
    def test_process_command_with_config(
        self, cli_runner, temp_events_file, temp_config_file
    ):
        """Test process command with custom config."""
        result = cli_runner.invoke(
            app,
            [
                "process",
                str(temp_events_file),
                "--config",
                str(temp_config_file),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        assert "📋 Configuration loaded:" in result.stdout
        assert "Slice duration: 1000ms" in result.stdout  # From temp config

    def test_process_command_nonexistent_file(self, cli_runner):
        """Test process command with nonexistent input file."""
        result = cli_runner.invoke(app, ["process", "nonexistent.json"])
        assert result.exit_code != 0

    @freeze_time("2025-01-15T12:00:00Z")
    def test_process_command_help(self, cli_runner):
        """Test process command help."""
        result = cli_runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process events from input file" in result.stdout
        assert "INPUT_FILE" in result.stdout
        assert "--config" in result.stdout
        assert "--verbose" in result.stdout


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_invalid_json_file(self, cli_runner):
        """Test processing invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            result = cli_runner.invoke(app, ["process", str(temp_path)])
            assert result.exit_code == 1
        finally:
            temp_path.unlink()

    @freeze_time("2025-01-15T12:00:00Z")
    def test_invalid_event_structure(self, cli_runner):
        """Test processing file with invalid event structure."""
        invalid_events = [{"invalid_field": "invalid_value"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_events, f)
            temp_path = Path(f.name)

        try:
            result = cli_runner.invoke(app, ["process", str(temp_path)])
            assert result.exit_code == 1
        finally:
            temp_path.unlink()


@freeze_time("2025-01-15T12:00:00Z")
class TestTimeConsistency:
    """Test time-dependent behavior with frozen time."""

    def test_multiple_runs_consistent(self, cli_runner, temp_events_file):
        """Test that multiple runs produce consistent results."""
        result1 = cli_runner.invoke(
            app, ["process", str(temp_events_file), "--verbose"]
        )
        result2 = cli_runner.invoke(
            app, ["process", str(temp_events_file), "--verbose"]
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Both runs should process the same number of slices
        assert "📊 Processed" in result1.stdout
        assert "📊 Processed" in result2.stdout

    def test_config_consistency(self, cli_runner):
        """Test that config show is consistent across calls."""
        result1 = cli_runner.invoke(app, ["config", "--show"])
        result2 = cli_runner.invoke(app, ["config", "--show"])

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result1.stdout == result2.stdout
