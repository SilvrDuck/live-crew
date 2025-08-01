"""Dependency models for live-crew scheduler.

Based on specification in .vibes/live_crew_spec.md section 2.4
Defines CrewDep, EventDep, and discriminated union Dependency.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class CrewDep(BaseModel):
    """Dependency on another crew's completion.

    Args:
        type: Must be 'crew' for discriminated union
        crew: Name of the crew to depend on (alphanumeric + underscore)
        offset: Slice offset relative to current slice (default: -1 for previous slice)

    Note:
        Negative offsets refer to past slices, positive to future slices.
        Default offset of -1 means "wait for crew completion from previous slice".
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable dependencies
        extra="forbid",  # Strict validation
    )

    type: Literal["crew"]
    crew: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9_]+$", description="Crew name to depend on"
    )
    offset: int = Field(
        default=-1, description="Slice offset (negative=past, positive=future)"
    )


class EventDep(BaseModel):
    """Dependency on event occurrence.

    Args:
        type: Must be 'event' for discriminated union
        event: Name of the event kind to depend on (alphanumeric + underscore)
        offset: Slice offset relative to current slice (default: 0 for current slice)

    Note:
        Negative offsets refer to past slices, positive to future slices.
        Default offset of 0 means "wait for event in current slice".
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable dependencies
        extra="forbid",  # Strict validation
    )

    type: Literal["event"]
    event: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9_]+$", description="Event kind to depend on"
    )
    offset: int = Field(
        default=0, description="Slice offset (negative=past, positive=future)"
    )


# Discriminated union type for dependency resolution
Dependency = Annotated[Union[CrewDep, EventDep], Field(discriminator="type")]
