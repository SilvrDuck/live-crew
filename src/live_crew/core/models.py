"""Event and Action data models for live-crew.

Based on specification in .vibes/live_crew_spec.md section 2.1
Converted from dataclasses to Pydantic models for validation and serialization.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from live_crew.config.settings import get_config

# TypeVar for generic payload types (invariant is safer when both storing and retrieving)
PayloadT = TypeVar("PayloadT")

# Constants to avoid magic numbers and strings
MAX_KIND_LENGTH: Final = 50
MAX_STREAM_ID_LENGTH: Final = 100
DEFAULT_TTL_MS: Final = 5_000
MAX_TTL_MS: Final = 300_000
DEFAULT_STREAM_ID: Final = "default"


def validate_timestamp_field(timestamp: datetime, config=None) -> datetime:
    """Validate timestamp and ensure UTC timezone.

    Args:
        timestamp: Datetime to validate
        config: Optional config object. If None, uses global cached config.

    Returns:
        Validated timestamp in UTC

    Raises:
        ValueError: If timestamp is outside allowed window
    """
    cfg = config or get_config()

    # Convert naive datetime to UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC if not already
        timestamp = timestamp.astimezone(timezone.utc)

    # Skip validation if disabled
    if not cfg.event_validation.timestamp_validation_enabled:
        return timestamp

    now = datetime.now(timezone.utc)

    # Check future timestamps (configurable tolerance)
    if timestamp > now + timedelta(
        seconds=cfg.event_validation.future_tolerance_seconds
    ):
        raise ValueError(
            f"Timestamp cannot be more than {cfg.event_validation.future_tolerance_seconds}s in the future"
        )

    # Check past timestamps (configurable window)
    if timestamp < now - timedelta(days=cfg.event_validation.timestamp_window_days):
        raise ValueError(
            f"Timestamp cannot be more than {cfg.event_validation.timestamp_window_days} days in the past"
        )

    return timestamp


class Event(BaseModel, Generic[PayloadT]):
    """Generic event with timestamp, kind, stream_id, and payload.

    Args:
        ts: Event timestamp in UTC
        kind: Event type identifier (1-50 characters, alphanumeric + underscore)
        stream_id: Stream identifier (1-100 characters, alphanumeric + underscore/dash)
        payload: Event-specific data of type PayloadT

    Note:
        Events are immutable and optimized for NATS transport.
        Processing occurs within 500ms time slices.
    """

    model_config = ConfigDict(
        frozen=True,  # Required by spec for immutability
        extra="forbid",  # Strict validation - reject unknown fields at API boundaries
        arbitrary_types_allowed=True,  # Allow datetime and other types
    )

    ts: datetime = Field(
        description="Event timestamp in UTC", json_schema_extra={"format": "date-time"}
    )
    kind: str = Field(
        min_length=1,
        max_length=MAX_KIND_LENGTH,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Event type identifier",
    )
    stream_id: str = Field(
        min_length=1,
        max_length=MAX_STREAM_ID_LENGTH,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Stream identifier",
    )
    payload: PayloadT

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, timestamp: datetime) -> datetime:
        return validate_timestamp_field(timestamp)


class Action(BaseModel, Generic[PayloadT]):
    """Generic action with timestamp, kind, stream_id, payload, and TTL.

    Args:
        ts: Action timestamp in UTC
        kind: Action type identifier (1-50 characters, alphanumeric + underscore)
        stream_id: Stream identifier (1-100 characters, alphanumeric + underscore/dash)
        payload: Action-specific data of type PayloadT
        ttl_ms: Time-to-live in milliseconds (1-300,000ms, default 5,000ms)

    Note:
        Actions are immutable and serializable for NATS transport with a
        time-to-live specified in milliseconds. Default TTL is 5 seconds.
    """

    model_config = ConfigDict(
        frozen=True,  # Required by spec for immutability
        extra="forbid",  # Strict validation - reject unknown fields at API boundaries
        arbitrary_types_allowed=True,  # Allow datetime and other types
    )

    ts: datetime = Field(
        description="Action timestamp in UTC", json_schema_extra={"format": "date-time"}
    )
    kind: str = Field(
        min_length=1,
        max_length=MAX_KIND_LENGTH,
        pattern=r"^[a-zA-Z0-9_]+$",
        description="Action type identifier",
    )
    stream_id: str = Field(
        min_length=1,
        max_length=MAX_STREAM_ID_LENGTH,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Stream identifier",
    )
    payload: PayloadT
    ttl_ms: int = Field(
        default=DEFAULT_TTL_MS,
        gt=0,
        le=MAX_TTL_MS,
        description="Time-to-live in milliseconds",
    )

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, timestamp: datetime) -> datetime:
        return validate_timestamp_field(timestamp)

    @classmethod
    def from_event(
        cls,
        event: "Event[Any]",
        kind: str,
        payload: PayloadT,
        ttl_ms: int = DEFAULT_TTL_MS,
    ) -> "Action[PayloadT]":
        """Create action from an existing event with convenience defaults.

        Args:
            event: Source event to derive timestamp and stream_id from
            kind: Action type identifier
            payload: Action-specific data
            ttl_ms: Optional TTL override

        Returns:
            New Action instance with event's timestamp and stream_id
        """
        return cls(
            ts=event.ts,
            kind=kind,
            stream_id=event.stream_id,
            payload=payload,
            ttl_ms=ttl_ms,
        )

    @classmethod
    def create(
        cls,
        kind: str,
        payload: PayloadT,
        stream_id: str = DEFAULT_STREAM_ID,
        ttl_ms: int = DEFAULT_TTL_MS,
    ) -> "Action[PayloadT]":
        """Create action with current timestamp and convenience defaults.

        Args:
            kind: Action type identifier
            payload: Action-specific data
            stream_id: Stream identifier (defaults to "default")
            ttl_ms: Time-to-live in milliseconds

        Returns:
            New Action instance with current timestamp
        """
        return cls(
            ts=datetime.now(timezone.utc),
            kind=kind,
            stream_id=stream_id,
            payload=payload,
            ttl_ms=ttl_ms,
        )
