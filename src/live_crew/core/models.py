"""Event and Action data models for live-crew.

Based on specification in .vibes/live_crew_spec.md section 2.1
Converted from dataclasses to Pydantic models for validation and serialization.
"""

from datetime import datetime, timedelta, timezone
from typing import Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

# TypeVar for generic payload types (invariant is safer when both storing and retrieving)
PayloadT = TypeVar("PayloadT")

# Constants to avoid magic numbers
MAX_KIND_LENGTH: Final = 50
MAX_STREAM_ID_LENGTH: Final = 100
DEFAULT_TTL_MS: Final = 5_000
MAX_TTL_MS: Final = 300_000


def validate_timestamp_field(timestamp: datetime) -> datetime:
    """Validate timestamp and ensure UTC timezone."""
    # Convert naive datetime to UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        # Convert to UTC if not already
        timestamp = timestamp.astimezone(timezone.utc)

    # Prevent future timestamps (allow small clock skew)
    now = datetime.now(timezone.utc)
    if timestamp > now + timedelta(seconds=60):
        raise ValueError("Timestamp cannot be more than 60s in the future")

    # Prevent extremely old timestamps
    if timestamp < now - timedelta(days=30):
        raise ValueError("Timestamp cannot be more than 30 days in the past")

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
