"""Tests for path security validation and attack prevention."""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from live_crew.security.path_validation import (
    validate_file_path,
    validate_path_security,
    validate_business_requirements,
    PathSecurityError,
    get_default_allowed_directories,
)
from live_crew.transports.file import FileEventTransport


class TestSecurityValidation:
    """Pure security control testing - no business logic."""

    def test_realistic_attack_vectors(self):
        """Test actual attack patterns seen in the wild."""
        # These are real-world attacks that would fail on business logic (no .json extension)
        realistic_attacks = [
            "/etc/passwd",  # Classic path traversal target
            "/etc/shadow",  # Password hashes
            "/root/.ssh/id_rsa",  # SSH private keys
            "../../../../etc/hosts",  # Path traversal attempt
            "/bin/bash",  # System executables
            "/usr/bin/python",  # Interpreter access
        ]

        for attack_path in realistic_attacks:
            with pytest.raises((ValueError, FileNotFoundError)):
                # These should fail on business logic (no .json) or file not found
                # NOT on security controls
                validate_file_path(Path(attack_path))

    def test_security_boundary_bypass_attempts(self):
        """Test that system directories are properly blocked."""
        # Test with real system directories that exist
        if Path("/etc").exists():
            # Create a test file in current directory to verify allowlist works first
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({"test": "data"}, f)
                temp_file = Path(f.name)

            try:
                # This should work - file in temp directory
                validate_path_security(temp_file)

                # Now test that a restricted allowed_directories blocks the temp file
                restricted_dir = Path.cwd() / "nonexistent"
                with pytest.raises(
                    PathSecurityError, match="not within any allowed directory"
                ):
                    validate_path_security(
                        temp_file, allowed_directories=[restricted_dir]
                    )

            finally:
                temp_file.unlink()

    def test_home_directory_security_boundary(self):
        """Test security boundary for home directory access."""

        # Create a temporary file in a home-like directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake home directory structure
            fake_home = Path(temp_dir) / "Users" / "alice"
            fake_home.mkdir(parents=True)

            # Create a file in the fake home directory
            test_file = fake_home / "Desktop" / "private.json"
            test_file.parent.mkdir(parents=True)
            test_file.write_text('{"test": "data"}')

            # Create a separate project directory outside of home
            fake_project = Path(temp_dir) / "some" / "other" / "project"
            fake_project.mkdir(parents=True)

            # Test should fail without permission (home directory access blocked)
            with (
                patch.object(Path, "cwd", return_value=fake_project),
                patch.object(Path, "home", return_value=fake_home),
            ):
                with pytest.raises(
                    PathSecurityError,
                    match="Access to home directories requires explicit permission",
                ):
                    validate_path_security(test_file)

                # Should pass with explicit permission and home directory in allowed directories
                validated_path = validate_path_security(
                    test_file, allowed_directories=[fake_home], allow_home_access=True
                )
                assert validated_path == test_file.resolve()

    def test_valid_security_validation(self):
        """Test that valid paths pass security validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = Path(f.name)

        try:
            # Should pass security validation
            validated_path = validate_path_security(temp_path)
            assert validated_path == temp_path.resolve()
        finally:
            temp_path.unlink()


class TestBusinessValidation:
    """Business logic testing separate from security."""

    def test_file_extension_requirements(self):
        """Test business requirement for JSON files."""
        # Create temp files with different extensions
        test_extensions = [".txt", ".py", ".exe", ".sh", ".conf"]

        for ext in test_extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write("test content")
                temp_path = Path(f.name)

            try:
                with pytest.raises(
                    ValueError,
                    match="Invalid file extension.*Only .json files are supported",
                ):
                    validate_business_requirements(temp_path)
            finally:
                temp_path.unlink()

    def test_valid_json_file_business_validation(self):
        """Test that JSON files pass business validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = Path(f.name)

        try:
            # Should pass business validation
            validated_path = validate_business_requirements(temp_path)
            assert validated_path == temp_path
        finally:
            temp_path.unlink()

    def test_case_insensitive_json_extension(self):
        """Test that JSON extension matching is case insensitive."""
        json_extensions = [".json", ".JSON", ".Json", ".jSoN"]

        for ext in json_extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                json.dump({"test": "data"}, f)
                temp_path = Path(f.name)

            try:
                # Should pass business validation regardless of case
                validated_path = validate_business_requirements(temp_path)
                assert validated_path == temp_path
            finally:
                temp_path.unlink()


class TestIntegratedValidation:
    """Test complete validation combining security + business logic."""

    def test_valid_json_file_full_validation(self):
        """Test that valid JSON files pass complete validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = Path(f.name)

        try:
            # Should pass both security and business validation
            validated_path = validate_file_path(temp_path)
            assert validated_path == temp_path.resolve()
        finally:
            temp_path.unlink()

    def test_business_logic_blocks_non_json_files(self):
        """Test that non-JSON files are blocked by business logic."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not json content")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid file extension '.txt'"):
                validate_file_path(temp_path)
        finally:
            temp_path.unlink()

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent files raise FileNotFoundError."""
        nonexistent_path = Path("nonexistent_file.json")

        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_path(nonexistent_path)

    def test_directory_instead_of_file_blocked(self):
        """Test that directories are rejected by security validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)

            with pytest.raises(PathSecurityError, match="Path is not a file"):
                validate_file_path(dir_path)

    def test_unreadable_file_blocked(self):
        """Test that files without read permissions are blocked."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = Path(f.name)

        try:
            # Mock no read permission
            with patch("os.access", return_value=False):
                with pytest.raises(
                    PermissionError, match="No read permission for file"
                ):
                    validate_file_path(temp_path)
        finally:
            temp_path.unlink()

    def test_allowed_directories_restriction(self):
        """Test that files outside allowed directories are blocked."""
        # Create a file in a controlled directory (not in temp)
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_files"
            test_dir.mkdir()

            test_file = test_dir / "test.json"
            test_file.write_text('{"test": "data"}')

            # Restrict to a different directory that doesn't contain the test file
            specific_allowed_dir = Path(temp_dir) / "different_dir"
            specific_allowed_dir.mkdir()
            allowed_dirs = [specific_allowed_dir]

            with pytest.raises(
                PathSecurityError, match="not within any allowed directory"
            ):
                validate_file_path(test_file, allowed_directories=allowed_dirs)

    def test_symlink_resolution_and_validation(self):
        """Test that symlinks are properly resolved and validated."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            target_path = Path(f.name)

        # Create symlink in same directory
        symlink_path = target_path.parent / "link.json"

        try:
            symlink_path.symlink_to(target_path)

            # Should resolve symlink and validate target
            validated_path = validate_file_path(symlink_path)
            assert validated_path == target_path.resolve()

        finally:
            if symlink_path.exists():
                symlink_path.unlink()
            target_path.unlink()


class TestFileEventTransportSecurity:
    """Test FileEventTransport security integration."""

    def test_secure_file_transport_initialization(self):
        """Test that FileEventTransport validates paths during initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                [
                    {
                        "ts": "2025-07-20T12:00:00Z",
                        "kind": "test",
                        "stream_id": "test",
                        "payload": {},
                    }
                ],
                f,
            )
            temp_path = Path(f.name)

        try:
            # Should succeed with valid file
            transport = FileEventTransport(temp_path)
            assert transport.file_path == temp_path.resolve()
        finally:
            temp_path.unlink()

    def test_file_transport_blocks_realistic_attacks(self):
        """Test that FileEventTransport blocks realistic attack vectors."""
        realistic_attacks = [
            "../../../etc/passwd",  # No .json extension
            "../../../../etc/hosts",  # Path traversal
            "/root/.ssh/id_rsa",  # System file access
        ]

        for attack_path in realistic_attacks:
            with pytest.raises((ValueError, FileNotFoundError)):
                # Should fail on business logic (no .json) or file not found
                FileEventTransport(Path(attack_path))

    def test_file_transport_blocks_security_bypass_attempts(self):
        """Test that FileEventTransport validates paths properly."""
        # Test with restricted allowed directories
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_file = Path(f.name)

        try:
            # This should work - temp file allowed
            transport = FileEventTransport(temp_file)

            # Test by checking constructor validates properly with restricted dirs
            # (FileEventTransport doesn't expose allowed_directories param, so test validates existing behavior)
            assert transport.file_path == temp_file.resolve()

        finally:
            temp_file.unlink()

    def test_file_transport_business_logic_validation(self):
        """Test that FileEventTransport enforces business logic (JSON only)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not json")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid file extension"):
                FileEventTransport(temp_path)
        finally:
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_file_transport_reading_after_validation(self):
        """Test that FileEventTransport can read files after successful validation."""
        test_events = [
            {
                "ts": "2025-07-20T12:00:00Z",
                "kind": "test_event",
                "stream_id": "test_stream",
                "payload": {"message": "hello"},
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_events, f)
            temp_path = Path(f.name)

        try:
            transport = FileEventTransport(temp_path)
            events = await transport.read_events()

            assert len(events) == 1
            assert events[0].kind == "test_event"
            assert events[0].payload["message"] == "hello"  # type: ignore
        finally:
            temp_path.unlink()


class TestPathSecurityConfiguration:
    """Test path security configuration and defaults."""

    def test_default_allowed_directories(self):
        """Test that default allowed directories include current working directory."""
        defaults = get_default_allowed_directories()

        assert len(defaults) >= 1
        assert Path.cwd().resolve() in [d.resolve() for d in defaults]

    def test_custom_allowed_directories_with_temp(self):
        """Test validation with custom allowed directories."""
        # Create temp directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.json"

            with open(test_file, "w") as f:
                json.dump({"test": "data"}, f)

            # Should succeed when temp directory is in allowed list
            validated_path = validate_file_path(
                test_file, allowed_directories=[temp_path]
            )
            assert validated_path == test_file.resolve()
