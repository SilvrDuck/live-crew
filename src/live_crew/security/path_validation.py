"""Secure path validation utilities for file operations.

This module provides security validation for file paths to prevent:
- Directory traversal attacks (../../etc/passwd)
- Access to system files and directories
- Symlink-based attacks
- Unauthorized file access outside allowed directories
"""

import os
from pathlib import Path


class PathSecurityError(ValueError):
    """Raised when a path fails security validation."""

    pass


# System directories that should be blocked for security
# NOTE: This uses a hybrid whitelist-first + blacklist approach for defense in depth:
# 1. PRIMARY: Whitelist approach (allowed_directories) - only permit access to specified safe dirs
# 2. SECONDARY: Blacklist approach (below) - additional protection against known dangerous dirs
# This hybrid approach is security best practice, providing layered protection
SYSTEM_DIRECTORIES = {
    # Unix/Linux system directories
    "/etc",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/boot",
    "/dev",
    "/proc",
    "/sys",
    "/root",
    "/var/log",
    "/var/run",
    # macOS specific (but allow temp directories)
    "/System",
    "/Library",
    "/Applications",
    # Windows system directories
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\Users\\All Users",
    "C:\\ProgramData",
    "C:\\System Volume Information",
}


def validate_path_security(
    file_path: Path,
    allowed_directories: list[Path] | None = None,
    allow_home_access: bool = False,
) -> Path:
    """Security-only validation - no business logic mixing.

    Validates against security threats: path traversal, system directory access,
    permissions, and directory allowlists. Does NOT validate business requirements
    like file extensions.

    Args:
        file_path: Path to validate for security
        allowed_directories: List of directories that are allowed for access.
                           If None, defaults to current working directory and subdirectories.
        allow_home_access: Whether to allow access to user home directories

    Returns:
        Validated and resolved absolute path

    Raises:
        PathSecurityError: If path fails security validation
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # 1. Resolve the path to its absolute, canonical form first (fixes TOCTOU)
    try:
        resolved_path = file_path.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except (OSError, RuntimeError) as e:
        raise PathSecurityError(f"Cannot resolve path '{file_path}': {e}") from e

    # 2. Perform all checks on the RESOLVED path (prevents TOCTOU)
    if not resolved_path.is_file():
        raise PathSecurityError(f"Path is not a file: {resolved_path}")

    if not os.access(resolved_path, os.R_OK):
        raise PermissionError(f"No read permission for file: {resolved_path}")

    # 3. Securely check against prohibited system directories using pathlib
    resolved_system_dirs = {
        Path(p).resolve() for p in SYSTEM_DIRECTORIES if Path(p).exists()
    }
    for system_dir in resolved_system_dirs:
        if system_dir in resolved_path.parents:
            raise PathSecurityError(
                f"Access to system directory '{system_dir}' is not allowed: {resolved_path}. "
                f"System directories are blocked for security. If you absolutely trust this file, "
                f"consider copying it to an allowed directory instead."
            )

    # 4. Securely check home directory access using Path.home()
    if not allow_home_access:
        try:
            home_dir = Path.home().resolve()
            if home_dir in resolved_path.parents:
                # Allow if the cwd is a parent of the resolved path (e.g., running from a user's project folder)
                cwd = Path.cwd().resolve()
                if cwd not in resolved_path.parents:
                    raise PathSecurityError(
                        f"Access to home directories requires explicit permission: {resolved_path}. "
                        f"If you trust this file, use allow_home_access=True in your configuration."
                    )
        except (RuntimeError, ImportError):
            # Path.home() can fail in some environments (e.g., no HOME env var).
            # This is safer than falling back to pattern matching - just deny access
            raise PathSecurityError(
                f"Cannot determine home directory for security validation: {resolved_path}"
            )

    # 5. Robust allowlist validation using pathlib
    if allowed_directories is None:
        # Default: current working directory plus common temp directories for legitimate use
        allowed_directories = [Path.cwd()]
        # Add system temp directories that actually exist (legitimate production use)
        system_temp_dirs = [Path("/tmp"), Path("/var/tmp"), Path("/var/folders")]
        for temp_dir in system_temp_dirs:
            if temp_dir.exists():
                allowed_directories.append(temp_dir)

    resolved_allowed_dirs = [d.resolve() for d in allowed_directories]

    # Use is_relative_to for Python 3.9+, fallback for older versions
    try:
        path_is_allowed = any(
            resolved_path.is_relative_to(allowed_dir)
            for allowed_dir in resolved_allowed_dirs
        )
    except AttributeError:
        # Fallback for Python < 3.9
        path_is_allowed = any(
            allowed_dir in resolved_path.parents or resolved_path == allowed_dir
            for allowed_dir in resolved_allowed_dirs
        )

    if not path_is_allowed:
        allowed_dirs_str = ", ".join(str(d) for d in allowed_directories)
        raise PathSecurityError(
            f"File '{resolved_path}' is not within any allowed directory. "
            f"Allowed directories: {allowed_dirs_str}. "
            f"If you trust this file, add its directory to allowed_directories in your configuration."
        )

    return resolved_path


def validate_business_requirements(file_path: Path) -> Path:
    """Business logic validation - separate from security.

    Validates business requirements like file extensions, content types, etc.
    Should be called after security validation passes.

    Args:
        file_path: Path to validate for business requirements

    Returns:
        Path that meets business requirements

    Raises:
        ValueError: If path fails business validation
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Business requirement: only JSON files for event processing
    if file_path.suffix.lower() not in {".json"}:
        raise ValueError(
            f"Invalid file extension '{file_path.suffix}'. Only .json files are supported for event processing."
        )

    return file_path


def validate_file_path(
    file_path: Path,
    allowed_directories: list[Path] | None = None,
    allow_home_access: bool = False,
) -> Path:
    """Complete file path validation - security + business logic.

    Convenience function that performs both security and business validation.

    Args:
        file_path: Path to validate
        allowed_directories: List of directories that are allowed for access
        allow_home_access: Whether to allow access to user home directories

    Returns:
        Validated and resolved absolute path

    Raises:
        PathSecurityError: If path fails security validation
        ValueError: If path fails business validation
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    # First: Security validation (no business logic)
    validated_path = validate_path_security(
        file_path, allowed_directories, allow_home_access
    )

    # Second: Business logic validation
    return validate_business_requirements(validated_path)


def create_safe_working_directory() -> Path:
    """Create a safe working directory for file operations.

    Returns:
        Path to a secure temporary working directory
    """
    import tempfile

    # Create a secure temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="live_crew_"))
    return temp_dir


def get_default_allowed_directories() -> list[Path]:
    """Get default allowed directories for file operations.

    Returns:
        List of default safe directories (current working directory and subdirectories)
    """
    cwd = Path.cwd().resolve()
    return [cwd]
