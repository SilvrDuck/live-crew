"""Configuration loading and management for live-crew.

Based on specification in .vibes/live_crew_spec.md section 3.3
Provides configurable parameters for time slicing, heartbeats, and backends.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class LiveCrewConfig(BaseSettings):
    """Configuration model for live-crew with validation.

    Args:
        slice_ms: Time slice duration in milliseconds (1-10000ms)
        heartbeat_s: Heartbeat interval in seconds (1-300s)
        kv_backend: Key-value storage backend
        vector: Optional vector store configuration

    Note:
        Configuration is immutable once loaded for consistency.
    """

    model_config = SettingsConfigDict(
        yaml_file="live-config.yaml",
        env_prefix="LIVE_CREW_",
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
    vector: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional vector store configuration"
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
        """Customize settings sources - YAML file first, then env vars can override."""
        return (
            YamlConfigSettingsSource(settings_cls),
            env_settings,  # Environment variables can override YAML
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_config(config_path: Optional[Path] = None) -> LiveCrewConfig:
    """Load configuration using pydantic-settings.

    Args:
        config_path: Path to live-config.yaml file. If provided, overrides default.

    Returns:
        LiveCrewConfig instance with loaded or default values

    Raises:
        pydantic.ValidationError: If configuration values are invalid

    Note:
        Missing configuration file is not an error - defaults are used.
        Configuration sources in priority order:
        1. Environment variables (LIVE_CREW_ prefix)
        2. YAML file (live-config.yaml or provided path)
        3. Default values
    """
    if config_path is not None:
        # Override the default yaml_file if custom path provided
        class CustomPathConfig(LiveCrewConfig):
            model_config = SettingsConfigDict(
                yaml_file=str(config_path),
                env_prefix="LIVE_CREW_",
                frozen=True,
                extra="forbid",
            )

        return CustomPathConfig()

    # Use default configuration with standard yaml_file
    return LiveCrewConfig()
