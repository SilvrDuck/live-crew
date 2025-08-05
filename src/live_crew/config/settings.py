"""Configuration loading and management for live-crew.

Based on specification in .vibes/live_crew_spec.md section 3.3
Provides configurable parameters for time slicing, heartbeats, and backends.

Framework-friendly lazy-loaded configuration system with caching and override support.
"""

from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Context variable for YAML config path
_yaml_path: ContextVar[Path | None] = ContextVar("_yaml_path", default=None)


class EventValidationConfig(BaseModel):
    """Configuration for event timestamp and payload validation.

    Args:
        timestamp_validation_enabled: Enable timestamp validation checks
        timestamp_window_days: Maximum age of events in days
        future_tolerance_seconds: Maximum future timestamp tolerance in seconds
        strict_mode: Enable strict validation (reject unknown fields, etc.)
    """

    timestamp_validation_enabled: bool = Field(
        default=True, description="Enable timestamp validation checks"
    )
    timestamp_window_days: int = Field(
        default=30, ge=0, description="Maximum age of events in days"
    )
    future_tolerance_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Maximum future timestamp tolerance in seconds",
    )
    strict_mode: bool = Field(default=True, description="Enable strict validation mode")


class LiveCrewConfig(BaseSettings):
    """Configuration model for live-crew with validation.

    Args:
        slice_ms: Time slice duration in milliseconds (1-10000ms)
        heartbeat_s: Heartbeat interval in seconds (1-300s)
        kv_backend: Key-value storage backend
        vector: Optional vector store configuration

    Note:
        Configuration is immutable once loaded for consistency.
        Uses lazy loading - no file I/O until explicitly requested.
    """

    model_config = SettingsConfigDict(
        env_prefix="LIVE_CREW_",
        env_nested_delimiter="__",
        frozen=True,  # Immutable configuration
        extra="forbid",  # Strict validation - reject unknown fields
    )

    slice_ms: int = Field(
        default=500, gt=0, le=10000, description="Time slice duration in milliseconds"
    )
    heartbeat_s: int = Field(
        default=30, gt=0, le=300, description="Heartbeat interval in seconds"
    )
    kv_backend: Literal["jetstream", "redis", "memory"] = Field(
        default="jetstream", description="Key-value storage backend"
    )
    vector: Dict[str, Any] | None = Field(
        default=None, description="Optional vector store configuration"
    )
    event_validation: EventValidationConfig = Field(
        default_factory=EventValidationConfig,
        description="Event timestamp and payload validation configuration",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources - env vars win, YAML fills defaults."""
        path = _yaml_path.get()
        yaml_source = (
            YamlConfigSettingsSource(settings_cls, yaml_file=path) if path else None
        )

        # env wins, YAML fills defaults, then constructor kwargs and secrets
        sources = [env_settings]
        if yaml_source:
            sources.append(yaml_source)
        sources += [init_settings, file_secret_settings]
        return tuple(sources)


def load_config(config_path: str | Path | None = None) -> LiveCrewConfig:
    """Create a fresh config object from YAML + env vars.

    Args:
        config_path: Path to YAML config file. If provided, loads from that file.

    Returns:
        LiveCrewConfig instance with loaded or default values

    Raises:
        pydantic.ValidationError: If configuration values are invalid

    Note:
        Configuration sources in priority order:
        1. Environment variables (LIVE_CREW_ prefix) - highest priority
        2. YAML file (if provided)
        3. Default values - lowest priority
    """
    token = _yaml_path.set(Path(config_path) if config_path else None)
    try:
        return LiveCrewConfig()
    finally:
        _yaml_path.reset(token)


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> LiveCrewConfig:
    """Return a cached config object (lazy-loaded).

    Args:
        config_path: Path to YAML config file. Only used on first call.

    Returns:
        Cached LiveCrewConfig instance

    Note:
        This function provides the global singleton config for the framework.
        Use reload_config() to clear cache and reload with new settings.
    """
    return load_config(config_path)


def reload_config(config_path: str | Path | None = None) -> LiveCrewConfig:
    """Clear cache and reload settings.

    Args:
        config_path: Path to YAML config file for reloading

    Returns:
        Fresh LiveCrewConfig instance

    Note:
        Useful for tests and dynamic configuration changes.
        Clears the cached config and creates a new one.
    """
    get_config.cache_clear()
    return get_config(config_path)
